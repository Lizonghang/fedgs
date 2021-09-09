import argparse

DATASETS = ["sent140", "femnist", "shakespeare", "celeba", "synthetic", "reddit"]
SAMPLERS = ["random", "brute", "bayesian", "probability", "ga", "sgdd"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset",
                        help="dataset to be used;",
                        type=str,
                        choices=DATASETS,
                        default="femnist")
    parser.add_argument("-model",
                        help="neural network model to be used;",
                        type=str,
                        default="cnn")
    parser.add_argument("--num-rounds",
                        help="total rounds of external synchronizations;",
                        type=int,
                        default=500)
    parser.add_argument("--eval-every",
                        help="interval rounds for model evaluation;",
                        type=int,
                        default=1)
    parser.add_argument("--num-groups",
                        help="number of groups;",
                        type=int,
                        default=10)
    parser.add_argument("--clients-per-group",
                        help="number of clients selected in each group;",
                        type=int,
                        default=10)
    parser.add_argument("-sampler",
                        help="sampler to be used, can be random, brute, "
                             "bayesian, probability, ga and sgdd;",
                        type=str,
                        choices=SAMPLERS,
                        default="sgdd")
    parser.add_argument("--batch-size",
                        help="number of training samples in each batch;",
                        type=int,
                        default=32)
    parser.add_argument("--num-syncs",
                        help="number of internal synchronizations in each round;",
                        type=int,
                        default=50)
    parser.add_argument("-lr",
                        help="learning rate for local optimizers;",
                        type=float,
                        default=0.01)
    parser.add_argument("--seed",
                        help="seed for client selection and batch splitting;",
                        type=int,
                        default=0)
    parser.add_argument("--metrics-name",
                        help="name for metrics file;",
                        type=str,
                        default="metrics")
    parser.add_argument("--metrics-dir",
                        help="folder name for metrics files;",
                        type=str,
                        default="metrics")
    parser.add_argument("--log-dir",
                        help="folder name for log files;",
                        type=str,
                        default="logs")
    parser.add_argument("--log-rank",
                        help="identity of the container and log files;",
                        type=int,
                        default=0)
    parser.add_argument("--use-val-set",
                        help="use validation set;",
                        action="store_true")
    parser.add_argument("-ctx",
                        help="device for training, -1 for cpu and 0~7 for gpu;",
                        type=int,
                        default=-1)

    return parser.parse_args()
