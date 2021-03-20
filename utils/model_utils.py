import json
import os
import importlib
import mxnet as mx
from mxnet import nd, init
from collections import defaultdict

from baseline_constants import INPUT_SIZE


def batch_data(data, batch_size, seed):
    """Return batches of data as an iterator.
    Args:
        data: A dict := {"x": NDArray, "y": NDArray} (on one client).
        batch_size: Number of samples in a batch data.
        seed: The random number seed.
    Returns:
        batched_x: A batch of features of length: batch_size.
        batched_y: A batch of labels of length: batch_size.
    """
    data_x = data["x"]
    data_y = data["y"]

    epochs = 0
    while True:
        # randomly shuffle data
        mx.random.seed(seed + epochs)
        data_x = mx.random.shuffle(data_x)
        mx.random.seed(seed + epochs)
        data_y = mx.random.shuffle(data_y)

        # loop through mini-batches
        for i in range(0, len(data_x), batch_size):
            l = i
            r = min(i + batch_size, len(data_y))
            batched_x = data_x[l:r]
            batched_y = data_y[l:r]
            yield batched_x, batched_y

        epochs += 1


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """Parses data in given train and test data directories
    Assumes:
    - The data in the input directories are .json files with
        keys "users" and "user_data".
    - The set of train set users is the same as the set of test set users.
    Args:
        train_data_dir: Directories of train data.
        test_data_dir: Directories of test data.
    Returns:
        clients: List of client ids.
        groups: List of group ids; empty list if none found.
        train_data: Dictionary of train data.
        test_data: Dictionary of test data.
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def build_net(dataset, model_name, num_classes, ctx, seed=0, init=init.Xavier()):
    """Build neural network from the file {dataset}/{model_name}.py.
    Args:
        dataset: Name of dataset.
        model_name: Name of neural network model.
        num_classes: Number of classes to classify.
        ctx: The training context.
        init: The model weights initializer.
    Returns:
        net: The initialized neural network.
    """
    model_file = "%s/%s.py" % (dataset, model_name)
    if not os.path.exists(model_file):
        print("Please specify a valid model.")
    model_path = "%s.%s" % (dataset, model_name)
    mod = importlib.import_module(model_path)
    build_net_op = getattr(mod, "build_net")

    # build network
    net = build_net_op(num_classes)

    # initialize network
    mx.random.seed(seed)
    net.initialize(init=init, ctx=ctx)
    net(nd.random.uniform(shape=(1, *INPUT_SIZE), ctx=ctx))

    return net
