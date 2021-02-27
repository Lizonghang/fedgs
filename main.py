import importlib
import numpy as np
import os
import random
import mxnet as mx

import metrics.writer as metrics_writer

from client import Client
from server import TopServer, MiddleServer
from baseline_constants import MODEL_PARAMS
from utils.args import parse_args
from utils.model_utils import read_data


def main():
    args = parse_args()
    num_rounds = args.num_rounds
    eval_every = args.eval_every
    clients_per_group = args.clients_per_group
    ctx = mx.gpu(args.ctx) if args.ctx >= 0 else mx.cpu()

    log_dir = os.path.join(
        args.log_dir, args.dataset, str(args.log_rank))
    os.makedirs(log_dir, exist_ok=True)
    log_fn = "output.%i" % args.log_rank
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")

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
        return

    client_path = "%s.client_model" % args.dataset
    server_path = "%s.server_model" % args.dataset
    mod = importlib.import_module(client_path)
    ClientModel = getattr(mod, "ClientModel")
    mod = importlib.import_module(server_path)
    ServerModel = getattr(mod, "ServerModel")

    # learning rate, num_classes, and so on
    param_key = "%s.%s" % (args.dataset, args.model)
    model_params = MODEL_PARAMS[param_key]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    num_classes = model_params[1]

    # Create the shared client model
    client_model = ClientModel(
        args.seed, args.dataset, args.model, ctx, *model_params)

    # Create the shared middle server model
    middle_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    middle_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)

    # Create the top server model
    top_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    top_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)

    # Create clients
    clients, groups = setup_clients(client_model, args)
    _ = get_clients_info(clients)
    client_ids, client_groups, client_num_samples = _
    print("Total number of clients: %d" % len(clients),
          file=log_fp, flush=True)

    # Measure the global data distribution
    global_dist, _, _ = get_clients_dist(
        clients, display=True, max_num_clients=20, metrics_dir=args.metrics_dir)

    # Create middle servers
    middle_servers = setup_middle_servers(
        middle_server_model, middle_merged_update, groups)
    # [middle_servers[i].brief(log_fp) for i in range(args.num_groups)]
    print("Total number of middle servers: %d" % len(middle_servers),
          file=log_fp, flush=True)

    # Create the top server
    top_server = TopServer(
        top_server_model, top_merged_update, middle_servers)

    # Display initial status
    print("--- Random Initialization ---",
          file=log_fp, flush=True)
    stat_writer_fn = get_stat_writer_function(
        client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(
        0, top_server, client_num_samples, stat_writer_fn,
        args.use_val_set, log_fp)

    # Training simulation
    for r in range(1, num_rounds + 1):
        # Select clients
        top_server.select_clients(
            r, clients_per_group, args.sampler, global_dist, display=False,
            metrics_dir=args.metrics_dir, rand_per_group=args.rand_per_group)
        _ = get_clients_info(top_server.selected_clients)
        c_ids, c_groups, c_num_samples = _

        print("--- Round %d of %d: Training %d clients ---"
              % (r, num_rounds, len(c_ids)),
              file=log_fp, flush=True)

        # Simulate server model training on selected clients' data
        sys_metrics = top_server.train_model(r, args.num_syncs)
        sys_writer_fn(r, c_ids, sys_metrics, c_groups, c_num_samples)

        # Test model
        if r % eval_every == 0 or r == num_rounds:
            print_stats(
                r, top_server, client_num_samples, stat_writer_fn,
                args.use_val_set, log_fp)

    # Save the top server model
    top_server.save_model(log_dir)
    log_fp.close()


def create_clients(users, groups, train_data, test_data, model, args):
    # Randomly assign a group to each client, if groups are not given
    random.seed(args.seed)
    if len(groups) == 0:
        groups = [random.randint(0, args.num_groups - 1)
                  for _ in users]

    # Instantiate clients
    clients = [Client(args.seed, u, g, train_data[u],
                      test_data[u], model, args.batch_size)
               for u, g in zip(users, groups)]

    return clients


def group_clients(clients, num_groups):
    """Collect clients of each group into a list.
    Args:
        clients: List of all client objects.
        num_groups: Number of groups.
    Returns:
        groups: List of clients in each group.
    """
    groups = [[] for _ in range(num_groups)]
    for c in clients:
        groups[c.group].append(c)
    return groups


def setup_clients(model, args):
    """Load train, test data and instantiate clients.
    Args:
        model: The shared ClientModel object for all clients.
        args: Args entered from the command.
    Returns:
        clients: List of all client objects.
        groups: List of clients in each group.
    """
    eval_set = "test" if not args.use_val_set else "val"
    train_data_dir = os.path.join("data", args.dataset, "data", "train")
    test_data_dir = os.path.join("data", args.dataset, "data", eval_set)

    data = read_data(train_data_dir, test_data_dir)
    users, groups, train_data, test_data = data

    clients = create_clients(
        users, groups, train_data, test_data, model, args)

    groups = group_clients(clients, args.num_groups)

    return clients, groups


def get_clients_info(clients):
    """Returns the ids, groups and num_samples for the given clients.
    Args:
        clients: List of Client objects.
    Returns:
        ids: List of client_ids for the given clients.
        groups: Map of {client_id: group_id} for the given clients.
        num_samples: Map of {client_id: num_samples} for the given
            clients.
    """
    ids = [c.id for c in clients]
    groups = {c.id: c.group for c in clients}
    num_samples = {c.id: c.num_samples for c in clients}
    return ids, groups, num_samples


def get_clients_dist(
        clients, display=False, max_num_clients=20, metrics_dir="metrics"):
    """Return the global data distribution of all clients.
    Args:
        clients: List of Client objects.
        display: Visualize data distribution when set to True.
        max_num_clients: Maximum number of clients to plot.
    Returns:
        global_dist: List of num samples for each class.
        global_train_dist: List of num samples for each class in train set.
        global_test_dist: List of num samples for each class in test set.
    """
    global_train_dist = sum([c.train_sample_dist for c in clients])
    global_test_dist = sum([c.test_sample_dist for c in clients])
    global_dist = global_train_dist + global_test_dist

    if display:

        try:
            from metrics.visualization_utils import plot_clients_dist

            np.random.seed(0)
            rand_clients = np.random.choice(clients, max_num_clients)
            plot_clients_dist(clients=rand_clients,
                              global_dist=global_dist,
                              global_train_dist=global_train_dist,
                              global_test_dist=global_test_dist,
                              draw_mean=False,
                              metrics_dir=metrics_dir)

        except ModuleNotFoundError:
            pass

    return global_dist, global_train_dist, global_test_dist


def setup_middle_servers(server_model, merged_update, groups):
    """Instantiates middle servers based on given ServerModel objects.
    Args:
        server_model: A shared ServerModel object to store the middle
            server model.
        merged_update: A shared ServerModel object to merge updates
            from clients.
        groups: List of clients in each group.
    Returns:
        middle_servers: List of all middle servers.
    """
    num_groups = len(groups)
    middle_servers = [
        MiddleServer(g, server_model, merged_update, groups[g])
        for g in range(num_groups)]
    return middle_servers


def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples,
            partition, args.metrics_dir, "{}_{}_{}".format(
                args.metrics_name, "stat", args.log_rank))

    return writer_fn


def get_sys_writer_function(args):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples,
            "train", args.metrics_dir, "{}_{}_{}".format(
                args.metrics_name, "sys", args.log_rank))

    return writer_fn


def print_stats(num_round, server, num_samples, writer, use_val_set, log_fp=None):
    train_stat_metrics = server.test_model(set_to_use="train")
    print_metrics(
        train_stat_metrics, num_samples, prefix="train_", log_fp=log_fp)
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.test_model(set_to_use=eval_set)
    print_metrics(
        test_stat_metrics, num_samples, prefix="{}_".format(eval_set), log_fp=log_fp)
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix="", log_fp=None):
    """Prints weighted averages of the given metrics.
    Args:
        metrics: Dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: Dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print("%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g" \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)),
              file=log_fp, flush=True)


if __name__ == "__main__":
    main()
