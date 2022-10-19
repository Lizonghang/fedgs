import visualization_utils

SHOW_WEIGHTED = True  # show weighted accuracy instead of unweighted accuracy
PLOT_CLIENTS = True
PLOT_SET = "Test"  # "Test" or "Train"
PLOT_MOVE_AVG = False
WINDOW_SIZE = 5


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
        metrics, legend, use_set=PLOT_SET, weighted=SHOW_WEIGHTED,
        move_avg=PLOT_MOVE_AVG, window_size=WINDOW_SIZE, plot_stds=False)


def compare_loss_vs_round(metrics, legend):
    """Compare loss vs. round number across experiments."""
    visualization_utils.compare_loss_vs_round_number(
        metrics, legend, use_set=PLOT_SET, weighted=SHOW_WEIGHTED,
        move_avg=PLOT_MOVE_AVG, window_size=WINDOW_SIZE, plot_stds=False)


def compare_execution_time():
    """Compare execution time of Random, Brute, MC, Bayesian, GA
    and GBP-CS samplers."""
    samplers = ["Brute", "Bayesian", "GA", "MC", "GBP-CS", "Random"]
    exec_time = [978.7786, 7.9925, 0.9975, 0.1197, 0.0154, 0.0001]
    visualization_utils.compare_execution_time(samplers, exec_time)


def compare_distribution_divergence():
    """Compare the distribution divergence of Random, Brute, MC,
    Bayesian, GA and GBP-CS samplers."""
    samplers = ["Brute", "Bayesian", "GA", "MC", "GBP-CS", "Random"]
    dist_info = {
        "mean": [0.032, 0.047, 0.036, 0.040, 0.036, 0.083],
        "median": [0.030, 0.046, 0.036, 0.039, 0.034, 0.082],
        "std": [0.004, 0.009, 0.003, 0.005, 0.004, 0.010],
        "max": [0.038, 0.064, 0.041, 0.050, 0.042, 0.105],
        "min": [0.026, 0.035, 0.028, 0.031, 0.029, 0.072]
    }
    visualization_utils.compare_distribution_divergence(samplers, dist_info)


def compare_sampler_optim_curve():
    """Compare the optimization curves of Brute, MC, Bayesian, GA
    and GBP-CS samplers over time."""
    samplers = ["Brute", "Bayesian", "MC", "GA", "GBP-CS"]
    dist_info = {
        "GBP-CS": {"time": [0.003, 0.006, 0.009, 0.012, 0.015],
                 "dist": [0.037, 0.035, 0.031, 0.029, 0.029]},
        "MC": {"time": [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.033, 0.034,
                        0.041, 0.120],
               "dist": [0.073, 0.060, 0.058, 0.052, 0.047, 0.044, 0.043, 0.040,
                        0.035, 0.035]},
        "Brute": {"time": [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.008, 0.009,
                           0.013, 0.014, 0.015, 0.023, 0.040, 0.048, 0.083, 0.088,
                           0.194, 0.195, 0.254, 0.255, 0.294, 0.305, 0.383, 2.110,
                           2.127, 2.131, 2.225, 11.72, 23.14, 146.7, 978.8],
                  "dist": [0.079, 0.075, 0.070, 0.067, 0.066, 0.064, 0.063, 0.062,
                           0.059, 0.057, 0.056, 0.054, 0.051, 0.048, 0.046, 0.045,
                           0.045, 0.044, 0.044, 0.042, 0.040, 0.039, 0.038, 0.038,
                           0.037, 0.036, 0.034, 0.032, 0.029, 0.028, 0.028]},
        "Bayesian": {"time": [0.782, 1.563, 1.954, 2.735, 3.517, 5.080, 8.000],
                     "dist": [0.055, 0.052, 0.051, 0.047, 0.038, 0.035, 0.035]},
        "GA": {"time": [0.016, 0.022, 0.038, 0.047, 0.057, 1.000],
               "dist": [0.044, 0.042, 0.037, 0.036, 0.033, 0.033]}
    }
    visualization_utils.compare_sampler_optim_curve(samplers, dist_info)


def compare_gbp_cs_with_different_init_points():
    """Compare the optimization curves of GBP-CS with different
    initialization points."""
    init_strategy = ["Zero Init", "Rand Init", "MPInv Init"]
    dist_info = {
        "Zero Init": [0.123, 0.105, 0.075, 0.060, 0.047,
                      0.042, 0.036, 0.033, 0.030],
        "Random Init": [0.055, 0.046, 0.044],
        "MPInv Init": [0.037, 0.035, 0.031, 0.029]
    }
    visualization_utils.compare_gbp_cs_with_different_init_points(
        init_strategy, dist_info)


def plot_accuracy_surface_iterations_and_batchsize():
    """Plot the accuracy surface over different iterations and
    batch size."""
    xticks = [50, 30, 10]
    yticks = [64, 32, 16, 8]
    acc_map = [[0.857, 0.860, 0.860, 0.861],
               [0.850, 0.851, 0.852, 0.852],
               [0.791, 0.792, 0.792, 0.792]]
    visualization_utils.plot_accuracy_surface_iterations_and_batchsize(
        xticks, yticks, acc_map)


def plot_accuracy_surface_groups_and_clients():
    """Plot the accuracy surface over different number of groups
    and selected clients."""
    xticks = [20, 10, 5]
    yticks = [5, 10, 20, 40]
    acc_map = [[0.862, 0.864, 0.865, 0.867],
               [0.861, 0.860, 0.864, 0.865],
               [0.860, 0.861, 0.863, 0.864]]
    visualization_utils.plot_accuracy_surface_groups_and_clients(
        xticks, yticks, acc_map)


if __name__ == "__main__":
    metrics = visualization_utils.load_data(
        "metrics_stat_19.csv"
    )

    plot_acc_vs_round(*metrics)
    plot_loss_vs_round(*metrics)
    # plot_bytes_vs_round(*metrics)
    # plot_comp_vs_round(*metrics)

    stat_files = (
        "metrics_stat_21.csv",
        "metrics_stat_19.csv",
    )
    legend = (
        "FedGS(GBP-CS)",
        "FedAvg",
    )

    metrics = [visualization_utils.load_data(f)[0]
               for f in stat_files]

    compare_accuracy_vs_round(metrics, legend)
    compare_loss_vs_round(metrics, legend)
    compare_execution_time()
    compare_distribution_divergence()
    compare_sampler_optim_curve()
    compare_gbp_cs_with_different_init_points()
    plot_accuracy_surface_iterations_and_batchsize()
    plot_accuracy_surface_groups_and_clients()
