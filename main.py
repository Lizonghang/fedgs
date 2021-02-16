import importlib
import numpy as np
import os
import random
import mxnet as mx

import metrics.writer as metrics_writer

from client import Client
from server import TopServer, MidServer
from baseline_constants import MODEL_PARAMS
from utils.args import parse_args
from utils.model_utils import read_data


def main():
    args = parse_args()
    num_rounds = args.num_rounds
    eval_every = args.eval_every
    clients_per_round = args.clients_per_round
    ctx = mx.gpu(args.ctx) if args.ctx >= 0 else mx.cpu()

    log_dir = os.path.join(
        args.log_dir, args.dataset, str(args.log_rank))
    os.makedirs(log_dir, exist_ok=True)
    log_fn = "output.%i" % args.log_rank
    log_file = os.path.join(log_dir, log_fn)
    # log_fp = open(log_file, "w+")
    log_fp = None

    # Set the random seed, affects client sampling and batching
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    mx.random.seed(123 + args.seed)

    # Import the client model and server model
    client_path = "%s/client_model.py" % args.dataset
    server_path = "%s/server_model.py" % args.dataset
    if not os.path.exists(client_path) \
            or not os.path.exists(server_path):
        print("Please specify a valid dataset.",
              file=log_fp, flush=True)
    client_path = "%s.client_model" % args.dataset
    server_path = "%s.server_model" % args.dataset
    mod = importlib.import_module(client_path)
    ClientModel = getattr(mod, "ClientModel")
    mod = importlib.import_module(server_path)
    ServerModel = getattr(mod, "ServerModel")

    # model params (hyper)
    param_key = "%s.%s" % (args.dataset, args.model)
    model_params = MODEL_PARAMS[param_key]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    num_classes = model_params[1]

    # Create client model
    client_model = ClientModel(
        args.seed, args.dataset, args.model, ctx, args.count_ops, *model_params)

    # Create middle server model
    mid_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    mid_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)

    # Create top server model
    top_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    top_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)

    # Create clients
    clients, groups = setup_clients(
        args.dataset, client_model, args.use_val_set, args.num_groups)
    _ = get_clients_info(clients)
    client_ids, client_groups, client_num_samples = _
    print("Total number of clients: %d" % len(clients),
          file=log_fp, flush=True)

    # Create the top server
    top_server = TopServer(top_server_model, top_merged_update)

    # Create middle servers
    mid_servers = setup_middle_servers(
        mid_server_model, mid_merged_update, groups)
    [mid_servers[i].brief(log_fp) for i in range(args.num_groups)]

    # Display initial status
    print("--- Random Initialization ---",
          file=log_fp, flush=True)
    stat_writer_fn = get_stat_writer_function(
        client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(
        0, top_server, clients, client_num_samples,
        stat_writer_fn, args.use_val_set, log_fp)

    # log_fp.close()


def create_clients(users, groups, train_data, test_data, model, num_groups):
    if len(groups) == 0:
        groups = [random.randint(0, num_groups-1) for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model)
               for u, g in zip(users, groups)]
    return clients


def group_clients(clients, num_groups):
    groups = [[] for _ in range(num_groups)]
    for c in clients:
        groups[c.group].append(c)
    return groups


def setup_clients(dataset, model=None, use_val_set=False, num_groups=10):
    """Instantiates clients based on given train and test data directories.
    Return:
        all_clients: list of Client objects.
    """
    eval_set = "test" if not use_val_set else "val"
    train_data_dir = os.path.join("data", dataset, "data", "train")
    test_data_dir = os.path.join("data", dataset, "data", eval_set)

    data = read_data(train_data_dir, test_data_dir)
    users, groups, train_data, test_data = data

    clients = create_clients(users, groups, train_data, test_data, model, num_groups)
    groups = group_clients(clients, num_groups)
    return clients, groups


def get_clients_info(clients):
    """Returns the ids, hierarchies and num_samples for the given clients.
    Args:
        clients: list of Client objects.
    """
    ids = [c.id for c in clients]
    groups = {c.id: c.group for c in clients}
    num_samples = {c.id: c.num_samples for c in clients}
    return ids, groups, num_samples


def setup_middle_servers(server_model, merged_update, groups):
    num_groups = len(groups)
    mid_servers = [MidServer(g, server_model, merged_update, groups[g])
                   for g in range(num_groups)]
    return mid_servers


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition,
            args.metrics_dir, "{}_{}".format(args.metrics_name, "stat"))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, "train",
            args.metrics_dir, "{}_{}".format(args.metrics_name, "sys"))

    return writer_fn


def print_stats(num_round, server, clients, num_samples, writer, use_val_set, log_fp=None):
    train_stat_metrics = server.test_model(clients, set_to_use="train")
    print_metrics(
        train_stat_metrics, num_samples, prefix="train_", log_fp=log_fp)
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(
        test_stat_metrics, num_samples, prefix="{}_".format(eval_set), log_fp=log_fp)
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix="", log_fp=None):
    """Prints weighted averages of the given metrics.
    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)),
              file=log_fp, flush=True)


if __name__ == "__main__":
    main()
