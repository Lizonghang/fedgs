"""Tools to visualize metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from mxnet import nd

from decimal import Decimal

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(models_dir)

from baseline_constants import (
    ACCURACY_KEY,
    BYTES_READ_KEY,
    BYTES_WRITTEN_KEY,
    CLIENT_ID_KEY,
    LOCAL_COMPUTATIONS_KEY,
    NUM_ROUND_KEY,
    NUM_SAMPLES_KEY)

title_fontsize = 14
legend_fontsize = 13
label_fontsize = 14
tick_fontsize = 13


def load_data(stat_metrics_file, sys_metrics_file):
    """Loads the data from the given stat_metric and sys_metric files."""
    stat_metrics = pd.read_csv(stat_metrics_file) if stat_metrics_file else None
    sys_metrics = pd.read_csv(sys_metrics_file) if sys_metrics_file else None

    if stat_metrics is not None:
        stat_metrics.sort_values(by=NUM_ROUND_KEY, inplace=True)
    if sys_metrics is not None:
        sys_metrics.sort_values(by=NUM_ROUND_KEY, inplace=True)

    return stat_metrics, sys_metrics


def _set_plot_properties(properties):
    """Set plt properties."""
    if "xlim" in properties:
        plt.xlim(properties["xlim"])
    if "ylim" in properties:
        plt.ylim(properties["ylim"])
    if "xlabel" in properties:
        plt.xlabel(properties["xlabel"])
    if "ylabel" in properties:
        plt.ylabel(properties["ylabel"])


def plot_accuracy_vs_round_number(
        stat_metrics, use_set="Test", weighted=False,
        plot_stds=False, figsize=(6, 4.5),  **kwargs):
    """Plot the average accuracy vs the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        weighted: Whether the average across clients should be weighted by number of
            test samples.
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    plt.figure(figsize=figsize)
    title_weighted = "Weighted" if weighted else "Unweighted"
    plt.title("%s Accuracy vs Round Number (%s)" % (use_set, title_weighted),
              fontsize=title_fontsize)
    stat_metrics = stat_metrics.query("set=='%s'" % use_set.lower())

    if weighted:
        accuracies = stat_metrics.groupby(NUM_ROUND_KEY)\
            .apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
        accuracies = accuracies.reset_index(name=ACCURACY_KEY)

        stds = stat_metrics.groupby(NUM_ROUND_KEY)\
            .apply(_weighted_std, ACCURACY_KEY, NUM_SAMPLES_KEY)
        stds = stds.reset_index(name=ACCURACY_KEY)
    else:
        accuracies = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).mean()
        stds = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).std()

    if plot_stds:
        plt.errorbar(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], stds[ACCURACY_KEY])
    else:
        plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY])

    percentile_10 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.1)
    percentile_50 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.5)
    percentile_90 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.9)

    plt.plot(percentile_10[NUM_ROUND_KEY], percentile_10[ACCURACY_KEY], linestyle=":")
    plt.plot(percentile_50[NUM_ROUND_KEY], percentile_50[ACCURACY_KEY], linestyle=":")
    plt.plot(percentile_90[NUM_ROUND_KEY], percentile_90[ACCURACY_KEY], linestyle=":")

    plt.legend(["Mean", "10th percentile", "50th percentile", "90th percentile"],
               loc="lower right", fontsize=legend_fontsize)

    plt.ylabel("%s Accuracy" % use_set, fontsize=label_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    _set_plot_properties(kwargs)
    plt.show()


def plot_loss_vs_round_number(
        stat_metrics, use_set="Test", weighted=False,
        plot_stds=False, figsize=(6, 4.5),  **kwargs):
    """Plot the average loss vs the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        weighted: Whether the average across clients should be weighted by number of
            test samples.
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    plt.figure(figsize=figsize)
    title_weighted = "Weighted" if weighted else "Unweighted"
    plt.title("%s Loss vs Round Number (%s)" % (use_set, title_weighted),
              fontsize=title_fontsize)
    stat_metrics = stat_metrics.query("set=='%s'" % use_set.lower())

    if weighted:
        losses = stat_metrics.groupby(NUM_ROUND_KEY)\
            .apply(_weighted_mean, "loss", NUM_SAMPLES_KEY)
        losses = losses.reset_index(name="loss")

        stds = stat_metrics.groupby(NUM_ROUND_KEY)\
            .apply(_weighted_std, "loss", NUM_SAMPLES_KEY)
        stds = stds.reset_index(name="loss")
    else:
        losses = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).mean()
        stds = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).std()

    if plot_stds:
        plt.errorbar(losses[NUM_ROUND_KEY], losses["loss"], stds["loss"])
    else:
        plt.plot(losses[NUM_ROUND_KEY], losses["loss"])

    percentile_10 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.1)
    percentile_50 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.5)
    percentile_90 = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.9)

    plt.plot(percentile_10[NUM_ROUND_KEY], percentile_10["loss"], linestyle=":")
    plt.plot(percentile_50[NUM_ROUND_KEY], percentile_50["loss"], linestyle=":")
    plt.plot(percentile_90[NUM_ROUND_KEY], percentile_90["loss"], linestyle=":")

    plt.legend(["Mean", "10th percentile", "50th percentile", "90th percentile"],
               loc="upper right", fontsize=legend_fontsize)

    plt.ylabel("%s Loss" % use_set, fontsize=label_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 500))
    plt.ylim(bottom=0)
    _set_plot_properties(kwargs)
    plt.show()


def _weighted_mean(df, metric_name, weight_name):
    d = df[metric_name]
    w = df[weight_name]
    try:
        return (w * d).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def _weighted_std(df, metric_name, weight_name):
    d = df[metric_name]
    w = df[weight_name]
    try:
        weigthed_mean = (w * d).sum() / w.sum()
        return (w * ((d - weigthed_mean) ** 2)).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def plot_accuracy_vs_round_number_per_client(
        stat_metrics, sys_metrics, max_num_clients, use_set="Test",
        figsize=(10, 10), max_name_len=10, **kwargs):
    """Plot the clients' accuracy vs the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        sys_metrics: pd.DataFrame as written by writer.py. Allows us to know which client actually
            performed training in each round. If None, then no indication is given of when was
            each client trained.
        max_num_clients: Maximum number of clients to plot.
        figsize: Size of the plot as specified by plt.figure().
        max_name_len: Maximum length for a client's id.
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    clients = stat_metrics[CLIENT_ID_KEY].unique()[:max_num_clients]
    cmap = plt.get_cmap("jet_r")
    plt.figure(figsize=figsize)
    stat_metrics = stat_metrics.query("set=='%s'" % use_set.lower())

    for i, c in enumerate(clients):
        color = cmap(float(i) / len(clients))
        c_accuracies = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
        plt.plot(c_accuracies[NUM_ROUND_KEY], c_accuracies[ACCURACY_KEY], color=color)

    plt.suptitle("%s Accuracy vs Round Number (%s clients)" % (use_set, max_num_clients),
              fontsize=title_fontsize)
    plt.title("Dots indicate that this client was trained at that round.",
              fontsize=title_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.ylabel("%s Accuracy" % use_set, fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 500))
    plt.ylim((0, 1))

    labels = stat_metrics[[CLIENT_ID_KEY, NUM_SAMPLES_KEY]].drop_duplicates()
    labels = labels.loc[labels[CLIENT_ID_KEY].isin(clients)]
    labels = ["%s, %d" % (row[CLIENT_ID_KEY][:max_name_len], row[NUM_SAMPLES_KEY])
              for _, row in labels.iterrows()]
    plt.legend(labels, title="client id, num_samples",
               loc="lower right", fontsize=legend_fontsize)

    # Plot moments in which the clients were actually used for training.
    # To do this, we need the system metrics (to know which client actually
    # performed training in each round).
    if sys_metrics is not None:
        for i, c in enumerate(clients[:max_num_clients]):
            color = cmap(float(i) / len(clients))
            c_accuracies = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
            c_computation = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
            c_join = pd.merge(c_accuracies, c_computation, on=NUM_ROUND_KEY, how="inner")
            if not c_join.empty:
                plt.plot(
                    c_join[NUM_ROUND_KEY],
                    c_join[ACCURACY_KEY],
                    linestyle="None",
                    marker=".",
                    color=color,
                    markersize=18)

    _set_plot_properties(kwargs)
    plt.show()


def plot_loss_vs_round_number_per_client(
        stat_metrics, sys_metrics, max_num_clients, use_set="Test",
        figsize=(10, 10), max_name_len=10, **kwargs):
    """Plot the clients' loss vs the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        sys_metrics: pd.DataFrame as written by writer.py. Allows us to know which client actually
            performed training in each round. If None, then no indication is given of when was
            each client trained.
        max_num_clients: Maximum number of clients to plot.
        figsize: Size of the plot as specified by plt.figure().
        max_name_len: Maximum length for a client's id.
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    clients = stat_metrics[CLIENT_ID_KEY].unique()[:max_num_clients]
    cmap = plt.get_cmap("jet_r")
    plt.figure(figsize=figsize)
    stat_metrics = stat_metrics.query("set=='%s'" % use_set.lower())

    for i, c in enumerate(clients):
        color = cmap(float(i) / len(clients))
        c_losses = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
        plt.plot(c_losses[NUM_ROUND_KEY], c_losses["loss"], color=color)

    plt.suptitle("%s Loss vs Round Number (%s clients)" % (use_set, max_num_clients),
              fontsize=title_fontsize)
    plt.title("Dots indicate that this client was trained at that round.",
              fontsize=title_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.ylabel("%s Loss" % use_set, fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 500))
    plt.ylim(bottom=0)

    labels = stat_metrics[[CLIENT_ID_KEY, NUM_SAMPLES_KEY]].drop_duplicates()
    labels = labels.loc[labels[CLIENT_ID_KEY].isin(clients)]
    labels = ["%s, %d" % (row[CLIENT_ID_KEY][:max_name_len], row[NUM_SAMPLES_KEY])
              for _, row in labels.iterrows()]
    plt.legend(labels, title="client id, num_samples",
               loc="upper right", fontsize=legend_fontsize)

    # Plot moments in which the clients were actually used for training.
    # To do this, we need the system metrics (to know which client actually
    # performed training in each round).
    if sys_metrics is not None:
        for i, c in enumerate(clients[:max_num_clients]):
            color = cmap(float(i) / len(clients))
            c_losses = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
            c_computation = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
            c_join = pd.merge(c_losses, c_computation, on=NUM_ROUND_KEY, how="inner")
            if not c_join.empty:
                plt.plot(
                    c_join[NUM_ROUND_KEY],
                    c_join["loss"],
                    linestyle="None",
                    marker=".",
                    color=color,
                    markersize=18)

    _set_plot_properties(kwargs)
    plt.show()


def plot_bytes_written_and_read(
        sys_metrics, rolling_window=10, figsize=(6, 4.5), **kwargs):
    """Plot the cumulative sum of the bytes pushed and pulled by all clients.

    Args:
        sys_metrics: pd.DataFrame as written by writer.py.
        rolling_window: Number of previous rounds to consider in the cumulative sum.
        figsize: Size of the plot as specified by plt.figure().
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    plt.figure(figsize=figsize)

    server_metrics = sys_metrics.groupby(NUM_ROUND_KEY, as_index=False).sum()
    rounds = server_metrics[NUM_ROUND_KEY]
    server_metrics = server_metrics.rolling(
        rolling_window, on=NUM_ROUND_KEY, min_periods=1).sum()
    plt.plot(rounds, server_metrics["bytes_written"], alpha=0.7)
    plt.plot(rounds, server_metrics["bytes_read"], alpha=0.7)

    plt.title("Bytes Pushed and Pulled by Clients vs Round Number\n",
              fontsize=title_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.ylabel("Bytes", fontsize=label_fontsize)
    plt.legend(["Bytes Pushed", "Bytes Pulled"], loc="upper left",
               fontsize=legend_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 500))
    plt.ylim(bottom=0)
    _set_plot_properties(kwargs)
    plt.show()


def plot_client_computations_vs_round_number(
        sys_metrics, aggregate_window=20, max_num_clients=20,
        figsize=(10, 8), max_name_len=10, range_rounds=None):
    """Plot the clients' local computations against round number.

    Args:
        sys_metrics: pd.DataFrame as written by writer.py.
        aggregate_window: Number of rounds that are aggregated. e.g. If set to 20, then
            rounds 0-19, 20-39, etc. will be added together.
        max_num_clients: Maximum number of clients to plot.
        figsize: Size of the plot as specified by plt.figure().
        max_name_len: Maximum length for a client"s id.
        range_rounds: Tuple representing the range of rounds to be plotted. The rounds
            are subsampled before aggregation. If None, all rounds are considered.
    """
    plt.figure(figsize=figsize)

    num_rounds = sys_metrics[NUM_ROUND_KEY].max()
    clients = sys_metrics[CLIENT_ID_KEY].unique()[:max_num_clients]

    comp_matrix = []
    matrix_keys = [c[:max_name_len] for c in clients]

    for c in clients:
        client_rows = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
        client_rows = client_rows.groupby(NUM_ROUND_KEY, as_index=False).sum()
        client_computations = [0 for _ in range(num_rounds)]

        for i in range(num_rounds):
            computation_row = client_rows.loc[client_rows[NUM_ROUND_KEY] == i]
            if not computation_row.empty:
                client_computations[i] = computation_row.iloc[0][LOCAL_COMPUTATIONS_KEY]
        comp_matrix.append(client_computations)

    if range_rounds:
        assert range_rounds[0] >= 0 and range_rounds[1] > 0
        assert range_rounds[0] <= range_rounds[1]
        assert range_rounds[0] < len(comp_matrix[0]) \
               and range_rounds[1] < len(comp_matrix[0]) + 1
        assert range_rounds[1] - range_rounds[0] >= aggregate_window
        new_comp_matrix = []
        for i in range(len(comp_matrix)):
            new_comp_matrix.append(comp_matrix[i][range_rounds[0]:range_rounds[1]])
        comp_matrix = new_comp_matrix

    agg_comp_matrix = []
    for c_comp_vals in comp_matrix:
        num_rounds = len(c_comp_vals)
        agg_c_comp_vals = []
        for i in range(num_rounds // aggregate_window):
            agg_c_comp_vals.append(
                np.sum(c_comp_vals[i * aggregate_window:(i + 1) * aggregate_window]))
        agg_comp_matrix.append(agg_c_comp_vals)

    plt.title(
        "Total Client Computations (FLOPs) vs. Round Number (x %d)" % aggregate_window,
        fontsize=title_fontsize)
    im = plt.imshow(agg_comp_matrix)
    plt.yticks(range(len(matrix_keys)), matrix_keys)
    plt.colorbar(im, fraction=0.02, pad=0.01)
    plt.show()


def get_longest_flops_path(sys_metrics):
    """Print the largest amount of flops required to complete training.
    To calculate this metric, we:
        1. For each round, pick the client that required the largest amount
            of local training.
        2. Sum the FLOPS from the clients picked in step 1 across rounds.
    Args:
        sys_metrics: pd.DataFrame as written by writer.py.
    """
    num_rounds = sys_metrics[NUM_ROUND_KEY].max()
    clients = sys_metrics[CLIENT_ID_KEY].unique()

    comp_matrix = []

    for c in clients:
        client_rows = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
        client_rows = client_rows.groupby(NUM_ROUND_KEY, as_index=False).sum()
        client_computations = [0 for _ in range(num_rounds)]

        for i in range(num_rounds):
            computation_row = client_rows.loc[client_rows[NUM_ROUND_KEY] == i]
            if not computation_row.empty:
                client_computations[i] = computation_row.iloc[0][LOCAL_COMPUTATIONS_KEY]
        comp_matrix.append(client_computations)

    comp_matrix = np.asarray(comp_matrix)
    num_flops = np.sum(np.max(comp_matrix, axis=0))
    return "%.2E" % Decimal(num_flops.item())


def plot_clients_dist(clients=None,
                      global_dist=None,
                      global_train_dist=None,
                      global_test_dist=None,
                      draw_mean=False,
                      metrics_dir="metrics"):
    """Plot data distribution of given clients.
    Args:
        clients: List of Client objects.
        global_dist: List of num samples for each class.
        global_train_dist: List of num samples for each class in train set.
        global_test_dist: List of num samples for each class in test set.
    """
    plt.figure()
    plt.title("Data Distribution (%s Clients)" % len(clients),
              fontsize=title_fontsize)
    plt.xlabel("Class", fontsize=label_fontsize)
    plt.ylabel("Proportion", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 61))

    # plot clients' data distribution if given
    if clients is not None:
        num_classes = len(clients[0].train_sample_dist)
        class_list = range(num_classes)
        c_ids = []
        for c in clients:
            c_ids.append(c.id)
            c_train_dist_ = c.train_sample_dist
            c_train_dist_ = c_train_dist_ / c_train_dist_.sum()
            plt.plot(class_list, c_train_dist_, linestyle=":", linewidth=1)

    p0 = None
    if draw_mean and clients is not None:
        num_classes = len(clients[0].train_sample_dist)
        class_list = range(num_classes)
        c_mean_dist_ = sum([c.train_sample_dist for c in clients])
        c_mean_dist_ = c_mean_dist_ / c_mean_dist_.sum()
        p0, = plt.plot(
            class_list, c_mean_dist_, linestyle="--", linewidth=2, c="g")

    # plot the distribution of global train data
    p1 = None
    if global_train_dist is not None:
        num_classes = len(global_train_dist)
        class_list = range(num_classes)
        g_train_dist_ = global_train_dist / global_train_dist.sum()
        p1, = plt.plot(
            class_list, g_train_dist_, linestyle="--", linewidth=2, c="b")

    # plot the distribution of global test data
    p2 = None
    if global_test_dist is not None:
        num_classes = len(global_test_dist)
        class_list = range(num_classes)
        g_test_dist_ = global_test_dist / global_test_dist.sum()
        p2, = plt.plot(
            class_list, g_test_dist_, linestyle="--", linewidth=2, c="orange")

    # plot the distribution of global data
    p3 = None
    if global_dist is not None:
        num_classes = len(global_dist)
        class_list = range(num_classes)
        g_dist_ = global_dist / global_dist.sum()
        p3, = plt.plot(
            class_list, g_dist_, linestyle="-", linewidth=2, c="k")

    l = [[], []]
    if p0: l[0].append(p0); l[1].append("%s clients' mean dist" % len(clients));
    if p1: l[0].append(p1); l[1].append("global train dist");
    if p2: l[0].append(p2); l[1].append("global test dist");
    if p3: l[0].append(p3); l[1].append("global dist");

    plt.legend(*l, fontsize=10)
    plt.savefig(os.path.join(metrics_dir, "dist.png"))
