"""Tools to visualize metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from decimal import Decimal
from matplotlib import cm

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
text_fontsize = 12

colors = ("m", "orange", "b", "gray", "g", "purple", "pink")
hatches = ("x", "/", "\\", "-")


def load_data(stat_metrics_file, sys_metrics_file=None):
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
        use_set: Data used to plot.
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
        use_set: Data used to plot.
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


def _moving_average(df, window_size=4):
    tmp_ = df[:window_size - 1]
    rolled_df = df.rolling(window_size).mean()
    rolled_df[:window_size - 1] = tmp_
    return rolled_df


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
        use_set: Data used to plot.
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
        use_set: Data used to plot.
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


def compare_accuracy_vs_round_number(
        metrics, legend=None, use_set="Test", weighted=False, move_avg=False,
        window_size=4, plot_stds=False, figsize=(6, 4.5),  **kwargs):
    """Compare the average accuracy vs the round number.

    Args:
        metrics: List of pd.DataFrame objects to compare.
        legend: Legend text to be placed on the axes.
        use_set: Data used to plot.
        weighted: Whether the average across clients should be weighted by number of
            test samples.
        move_avg: Whether the accuracy is moving averaged.
        window_size: Size of the moving window.
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    plt.figure(figsize=figsize)
    title_weighted = "Weighted" if weighted else "Unweighted"
    plt.title("%s Accuracy vs Round Number (%s)" % (use_set, title_weighted),
              fontsize=title_fontsize)

    for stat_metrics in metrics:

        stat_metrics = stat_metrics.query("set=='%s'" % use_set.lower())

        if weighted:
            accuracies = stat_metrics.groupby(NUM_ROUND_KEY)\
                .apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
            if move_avg:
                accuracies = _moving_average(accuracies, window_size)
            accuracies = accuracies.reset_index(name=ACCURACY_KEY)


            stds = stat_metrics.groupby(NUM_ROUND_KEY)\
                .apply(_weighted_std, ACCURACY_KEY, NUM_SAMPLES_KEY)
            stds = stds.reset_index(name=ACCURACY_KEY)
        else:
            accuracies = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).mean()
            if move_avg:
                accuracies = _moving_average(accuracies, window_size)
            stds = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).std()

        if plot_stds:
            plt.errorbar(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], stds[ACCURACY_KEY])
        else:
            plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY])

    plt.legend(legend, loc="lower right", fontsize=legend_fontsize)

    plt.ylabel("%s Accuracy" % use_set, fontsize=label_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((60, 500))
    plt.ylim((0.7, 0.87))
    _set_plot_properties(kwargs)
    plt.show()


def compare_loss_vs_round_number(
        metrics, legend=None, use_set="Test", weighted=False, move_avg=False,
        window_size=4, plot_stds=False, figsize=(6, 4.5),  **kwargs):
    """Compare the average loss vs the round number.

    Args:
        metrics: List of pd.DataFrame objects to compare.
        legend: Legend text to be placed on the axes.
        use_set: Data used to plot.
        weighted: Whether the average across clients should be weighted by number of
            test samples.
        move_avg: Whether the loss is moving averaged.
        window_size: Size of the moving window.
        plot_stds: Whether to plot error bars corresponding to the std between users.
        figsize: Size of the plot as specified by plt.figure().
        kwargs: Arguments to be passed to _set_plot_properties.
    """
    plt.figure(figsize=figsize)
    title_weighted = "Weighted" if weighted else "Unweighted"
    plt.title("%s Loss vs Round Number (%s)" % (use_set, title_weighted),
              fontsize=title_fontsize)

    for stat_metrics in metrics:

        stat_metrics = stat_metrics.query("set=='%s'" % use_set.lower())

        if weighted:
            losses = stat_metrics.groupby(NUM_ROUND_KEY)\
                .apply(_weighted_mean, "loss", NUM_SAMPLES_KEY)
            if move_avg:
                losses = _moving_average(losses, window_size)
            losses = losses.reset_index(name="loss")

            stds = stat_metrics.groupby(NUM_ROUND_KEY)\
                .apply(_weighted_std, "loss", NUM_SAMPLES_KEY)
            stds = stds.reset_index(name="loss")
        else:
            losses = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).mean()
            if move_avg:
                losses = _moving_average(losses, window_size)
            stds = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False).std()

        if plot_stds:
            plt.errorbar(losses[NUM_ROUND_KEY], losses["loss"], stds["loss"])
        else:
            plt.plot(losses[NUM_ROUND_KEY], losses["loss"])

    plt.legend(legend, loc="upper right", fontsize=legend_fontsize)
    plt.ylabel("%s Loss" % use_set, fontsize=label_fontsize)
    plt.xlabel("Round Number", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((80, 500))
    plt.ylim((0.4, 1.2))
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
    plt.plot(rounds, server_metrics[BYTES_WRITTEN_KEY], alpha=0.7)
    plt.plot(rounds, server_metrics[BYTES_READ_KEY], alpha=0.7)

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
        draw_mean: Draw mean distribution of given clients.
        metrics_dir: Directory to save metrics files.
    """
    plt.figure(figsize=(7, 5))
    plt.title("Data Distribution (%s Clients)" % len(clients),
              fontsize=title_fontsize)
    plt.xlabel("Class", fontsize=label_fontsize)
    plt.ylabel("Proportion", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.xlim((0, 61))

    # plot clients' data distribution if given
    if clients is not None:
        num_classes = len(clients[0].next_train_batch_dist)
        class_list = range(num_classes)
        for c in clients:
            c_train_dist_ = c.next_train_batch_dist
            c_train_dist_ = c_train_dist_ / c_train_dist_.sum()
            plt.plot(class_list, c_train_dist_, linestyle=":", linewidth=1)

    p0 = None
    if draw_mean and clients is not None:
        num_classes = len(clients[0].next_train_batch_dist)
        class_list = range(num_classes)
        c_mean_dist_ = sum([c.next_train_batch_dist for c in clients])
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

    plt.legend(*l, fontsize=12)
    plt.savefig(os.path.join(metrics_dir, "dist.png"))
    plt.close()


def compare_execution_time(samplers, exec_time):
    """Plot execution time bars.
        Args:
            samplers: List of samplers used.
            exec_time: Execution Time for each sampler.
    """
    from brokenaxes import brokenaxes

    label_fontsize = 16
    tick_fontsize = 14
    plt.figure(figsize=(5, 4))

    bax = brokenaxes(ylims=((0, 8), (977, 980)), hspace=.3, despine=False)

    plt.xticks(range(len(samplers)), samplers, fontsize=tick_fontsize, rotation=15)
    plt.yticks(fontsize=tick_fontsize)

    bax.set_ylabel("Execution Time (s)", fontsize=label_fontsize)

    for i in range(len(samplers)):
        bax.bar(x=samplers[i], height=exec_time[i], align="center",
                color="w", edgecolor=colors[i])
        # bax.text(x=samplers[i], y=exec_time[i]+0.4, s=str(exec_time[i]),
        #          size=text_fontsize, horizontalalignment="center")

    plt.tight_layout()
    # plt.savefig(f"execution-time-compare.pdf")
    plt.show()


def compare_distribution_divergence(samplers, dist_info):
    """Plot the distribution divergence.
    Args:
        samplers: List of samplers used.
        dist_info: Information of distribution divergence, including:
            mean, median, std, max and min.
    """
    label_fontsize = 16
    tick_fontsize = 14
    text_fontsize = 12
    plt.figure(figsize=(5, 4))

    plt.ylabel("Dist Divergence (L2)", fontsize=label_fontsize)
    plt.xticks(range(len(samplers)), samplers, fontsize=tick_fontsize, rotation=15)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim((0, 0.112))

    for i in range(len(samplers)):
        mean_val_ = dist_info["mean"][i]
        max_val_ = dist_info["max"][i]
        min_val_ = dist_info["min"][i]
        # Draw mean bars
        plt.bar(x=samplers[i], height=mean_val_, align="center",
                # yerr=dist_info["std"][i],
                # error_kw=dict(elinewidth=1, ecolor=colors[i], capsize=5),
                color="w", edgecolor=colors[i])
        # Draw min and max error lines
        plt.plot((i-0.1, i+0.1), (max_val_, max_val_), color=colors[i])
        plt.plot((i-0.1, i+0.1), (min_val_, min_val_), color=colors[i])
        plt.plot((i, i), (min_val_, max_val_), color=colors[i])
        # Draw texts
        plt.text(x=i, y=min_val_-0.007, s=min_val_,
                 color=colors[i], horizontalalignment="center", fontsize=text_fontsize)
        plt.text(x=i, y=max_val_+0.002, s=max_val_,
                 color=colors[i], horizontalalignment="center", fontsize=text_fontsize)

    plt.tight_layout()
    # plt.savefig(f"dist-divergence-comparison.pdf")
    plt.show()


def compare_sampler_optim_curve(samplers, dist_info):
    """Plot the distance optimization curves.
    Args:
        samplers: List of samplers used.
        dist_info: Information of distance history of given samplers.
    """
    from brokenaxes import brokenaxes

    label_fontsize = 16
    legend_fontsize = 15
    plt.figure(figsize=(5, 4))

    bax = brokenaxes(xlims=((0.001, 0.015), (0.018, 1), (1.01, 8), (11, 979)),
                     width_ratios=[0.1, 0.1, 0.1, 0.1],
                     wspace=0, despine=False, d=0)

    bax.set_xlabel("Time (s)", fontsize=label_fontsize)
    bax.set_ylabel("Dist Divergence (L2)", fontsize=label_fontsize)
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75],
               [0, 0.007, 0.015, 0.5, 1, 4.5, 8])

    for i in range(len(samplers)):
        sampler = samplers[i]
        info = dist_info[sampler]
        bax.plot(info["time"], info["dist"],
                 color=colors[i], label=sampler)

    bax.plot((0, 200), (0.028, 0.028),
             c=colors[0], linestyle="--", linewidth=1)
    # plt.text(x=0.89, y=0.06, s="lower bound",
    #          color=colors[0], horizontalalignment="center",
    #          fontsize=text_fontsize)

    bax.plot((0.015, 0.015), (0.028, 0.079), c="k", linestyle=":")
    bax.plot((1, 1), (0.028, 0.079), c="k", linestyle=":")
    bax.plot((8, 8), (0.028, 0.079), c="k", linestyle=":")

    bax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    # plt.savefig(f"dist-optim-curve.pdf")
    plt.show()


def compare_gbp_cs_with_different_init_points(init_strategy, dist_info):
    """Plot the distance optimization curves of GBP-CS with different
    initialization points, including zero initialization, random
    initialization, and Moore–Penrose initialization.
    Args:
        init_strategy: Initialization strategies used in GBP-CS algorithm.
        dist_info: Information of distance history of different
            initialization strategies.
    """
    label_fontsize = 16
    tick_fontsize = 15
    legend_fontsize = 15
    text_fontsize = 15

    plt.figure(figsize=(5, 4))
    plt.xlabel("GBP-CS Iteration", fontsize=label_fontsize)
    plt.ylabel("Dist Divergence (L2)", fontsize=label_fontsize)
    plt.xlim(0, 8.5)
    plt.ylim(0.02, 0.13)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    for i in range(len(init_strategy)):
        strategy = init_strategy[i]
        info = dist_info[strategy]
        x = np.arange(len(info)).astype(dtype=np.str)
        plt.plot(x, info, color=colors[i], label=strategy)
        plt.plot(x[-1], info[-1],
                 marker="*", color=colors[i], markersize=10)
        plt.text(x[-1], info[-1]+0.004, info[-1], color=colors[i],
                 horizontalalignment="center", fontsize=text_fontsize)

    plt.plot((0, 8.5), (0.028, 0.028), color="k", linestyle=":")
    plt.text(x=4.25, y=0.021, s="Optimal Val: 0.028",
             horizontalalignment="center", fontsize=text_fontsize)

    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    # plt.savefig(f"dist-optim-init-curve.pdf")
    plt.show()


def plot_accuracy_surface_iterations_and_batchsize(xticks, yticks, acc_map):
    """Plot the accuracy surface of FedGS+GBP-CS over different
    iteration and batch size settings.
    Args:
        xticks: Ticks of axis x.
        yticks: Ticks of axis y.
        acc_map: The accuracy map.
    """
    label_fontsize = 16
    tick_fontsize = 14

    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Iterations Per Round \n $T$", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Batch Size \n $n$", fontsize=label_fontsize, labelpad=10)
    ax.set_zlabel("Accuracy", fontsize=label_fontsize, labelpad=10)
    plt.xticks(np.arange(len(xticks)), xticks, fontsize=tick_fontsize)
    plt.yticks(np.arange(len(yticks)), yticks, fontsize=tick_fontsize)

    x = np.arange(len(xticks))
    y = np.arange(len(yticks))
    x, y = np.meshgrid(x, y)
    z = np.array(acc_map).T

    surf = ax.plot_surface(x, y, z,
                           cmap=cm.coolwarm, linewidth=1, antialiased=False)

    position = fig.add_axes([0.05, 0.3, 0.06, 0.4])
    cb = plt.colorbar(surf, cax=position, shrink=0.5, aspect=5)
    cb.ax.tick_params(labelsize=tick_fontsize)

    # plt.savefig(f"acc-surface-batch-iter.pdf")
    plt.show()


def plot_accuracy_surface_groups_and_clients(xticks, yticks, acc_map):
    """Plot the accuracy surface of FedGS+GBP-CS over different
    number of groups and selected clients in each group.
    Args:
        xticks: Ticks of axis x.
        yticks: Ticks of axis y.
        acc_map: The accuracy map.
    """
    label_fontsize = 16
    tick_fontsize = 14

    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Num Groups \n $M$", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Selected Devices\n Per Group $L$", fontsize=label_fontsize, labelpad=10)
    ax.set_zlabel("Accuracy", fontsize=label_fontsize, labelpad=10)
    plt.xticks(np.arange(len(xticks)), xticks, fontsize=tick_fontsize)
    plt.yticks(np.arange(len(yticks)), yticks, fontsize=tick_fontsize)

    x = np.arange(len(xticks))
    y = np.arange(len(yticks))
    x, y = np.meshgrid(x, y)
    z = np.array(acc_map).T

    surf = ax.plot_surface(x, y, z,
                           cmap=cm.coolwarm, linewidth=1, antialiased=False)

    position = fig.add_axes([0.05, 0.3, 0.06, 0.4])
    cb = plt.colorbar(surf, cax=position, shrink=0.5, aspect=5)
    cb.ax.tick_params(labelsize=tick_fontsize)

    # plt.savefig(f"acc-surface-groups-devices.pdf")
    plt.show()
