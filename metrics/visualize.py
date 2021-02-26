import visualization_utils

SHOW_WEIGHTED = True  # show weighted accuracy instead of unweighted accuracy
PLOT_CLIENTS = True
PLOT_SET = "Test"  # "Test" or "Train"
stat_file = "metrics_stat.csv"  # change to None if desired
sys_file = "metrics_sys.csv"  # change to None if desired


def plot_acc_vs_round(stat_metrics, sys_metrics):
    """Plots accuracy vs. round number."""
    if stat_metrics is not None:
        visualization_utils.plot_accuracy_vs_round_number(
            stat_metrics, use_set=PLOT_SET, weighted=SHOW_WEIGHTED, plot_stds=False)
    if PLOT_CLIENTS and stat_metrics is not None:
        visualization_utils.plot_accuracy_vs_round_number_per_client(
            stat_metrics, sys_metrics, max_num_clients=10, use_set=PLOT_SET)


def plot_loss_vs_round(stat_metrics, sys_metrics):
    """Plots loss vs. round number."""
    if stat_metrics is not None:
        visualization_utils.plot_loss_vs_round_number(
            stat_metrics, use_set=PLOT_SET, weighted=SHOW_WEIGHTED, plot_stds=False)
    if PLOT_CLIENTS and stat_metrics is not None:
        visualization_utils.plot_loss_vs_round_number_per_client(
            stat_metrics, sys_metrics, max_num_clients=10, use_set=PLOT_SET)


def plot_bytes_vs_round(stat_metrics, sys_metrics):
    """Plots the cumulative sum of the bytes pushed and pulled by clients in
    the past rolling_window rounds versus the round number.
    """
    if stat_metrics is not None:
        visualization_utils.plot_bytes_written_and_read(
            sys_metrics, rolling_window=500)


def plot_comp_vs_round(stat_metrics, sys_metrics):
    visualization_utils.plot_client_computations_vs_round_number(
        sys_metrics, aggregate_window=10, max_num_clients=20, range_rounds=(0, 499))


def calc_longest_flops(stat_metrics, sys_metrics):
    print("Longest FLOPs path: %s" %
          visualization_utils.get_longest_flops_path(sys_metrics))


def compare_accuracy_vs_round(metrics, legend):
    """Compare accuracy vs. round number across experiments."""
    visualization_utils.compare_accuracy_vs_round_number(
        metrics, legend, use_set=PLOT_SET,
        weighted=SHOW_WEIGHTED, plot_stds=False)


def compare_loss_vs_round(metrics, legend):
    """Compare loss vs. round number across experiments."""
    visualization_utils.compare_loss_vs_round_number(
        metrics, legend, use_set=PLOT_SET,
        weighted=SHOW_WEIGHTED, plot_stds=False)


if __name__ == "__main__":
    # metrics = visualization_utils.load_data(stat_file, sys_file)
    # plot_acc_vs_round(*metrics)
    # plot_loss_vs_round(*metrics)
    # plot_bytes_vs_round(*metrics)
    # plot_comp_vs_round(*metrics)
    # calc_longest_flops(*metrics)

    stat_files = ("metrics_stat_9.csv", "metrics_stat_8.csv")
    legend = ("Exp 9", "Exp 8")

    metrics = [visualization_utils.load_data(f)[0]
               for f in stat_files]

    compare_accuracy_vs_round(metrics, legend)
    compare_loss_vs_round(metrics, legend)
