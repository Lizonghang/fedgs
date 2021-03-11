import argparse

DATASETS = ["sent140", "femnist", "shakespeare", "celeba", "synthetic", "reddit"]
SAMPLERS = ["random", "brute", "approx_iid", "bayesian", "probability", "ga"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset",
                        help="name of dataset;",
                        type=str,
                        choices=DATASETS,
                        default="femnist")
    parser.add_argument("-model",
                        help="name of model;",
                        type=str,
                        default="cnn")
    parser.add_argument("--num-rounds",
                        help="number of rounds to simulate;",
                        type=int,
                        default=500)
    parser.add_argument("--eval-every",
                        help="evaluate every _ rounds;",
                        type=int,
                        default=20)
    parser.add_argument("--num-groups",
                        help="number of groups;",
                        type=int,
                        default=10)
    parser.add_argument("--clients-per-group",
                        help="number of clients trained per group;",
                        type=int,
                        default=10)
    parser.add_argument("--rand-per-group",
                        help="number of randomly sampled clients per group;",
                        type=int,
                        default=2)
    parser.add_argument("-sampler",
                        help="name of sampler to be used;",
                        type=str,
                        choices=SAMPLERS,
                        default="random")
    parser.add_argument("--batch-size",
                        help="batch size when clients train on data;",
                        type=int,
                        default=5)
    parser.add_argument("--num-syncs",
                        help="number of local synchronizations in each group;",
                        type=int,
                        default=50)
    parser.add_argument("-lr",
                        help="learning rate for local optimizers;",
                        type=float,
                        default=0.01)
    parser.add_argument("--seed",
                        help="seed for random client sampling and batch splitting;",
                        type=int,
                        default=0)
    parser.add_argument("--metrics-name",
                        help="name for metrics file;",
                        type=str,
                        default="metrics")
    parser.add_argument("--metrics-dir",
                        help="dir for metrics file;",
                        type=str,
                        default="metrics")
    parser.add_argument("--log-dir",
                        help="dir for log file;",
                        type=str,
                        default="logs")
    parser.add_argument("--log-rank",
                        help="suffix identifier for log file;",
                        type=int,
                        default=0)
    parser.add_argument("--use-val-set",
                        help="use validation set;",
                        action="store_true")
    parser.add_argument("-ctx",
                        help="device for training, -1 for cpu and 0~3 for gpu;",
                        type=int,
                        default=-1)

    return parser.parse_args()
