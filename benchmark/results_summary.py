import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from collections import Counter

sns.set(font_scale=1.5)
sns.set_style("ticks")
sns.set_palette("colorblind")


# Maps internal model names to model names in plots and tables
MODEL_ALIASES = {
    "basicphaseae": "BasicPhaseAE",
    "cred": "CRED",
    "dppdetect": "DPP",
    "dpppickerp": "DPP",
    "dpppickers": "DPP",
    "eqtransformer": "EQTransformer",
    "gpd": "GPD-Org",
    "gpdpick": "GPD",
    "phasenet": "PhaseNet",
}
# Maps internal data names to model names in plots and tables
DATA_ALIASES = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "instance": "INSTANCE",
    "lendb": "LenDB",
    "neic": "NEIC",
    "scedc": "SCEDC",
    "stead": "STEAD",
}
# Provides classes for the data used for grouping the datasets
DATA_CLASSES = {
    "ethz": "regional",
    "geofon": "tele",
    "instance": "regional",
    "lendb": "regional",
    "neic": "tele",
    "scedc": "regional",
    "stead": "regional",
}


def main(base, cross, resampled):
    if not (base or cross or resampled):
        logging.warning("No task selected. exiting.")

    if base:
        results = pd.read_csv("results.csv")

        detect_missing_entries(results)
        print("Generating tables")
        results_tables(results, suffix="_gpd")
        results = results[results["model"] != "gpd"]
        results_tables(results)

        print("Generating plots")
        results_plots(results)

    if cross:
        results_cross = pd.read_csv("results_cross.csv")
        results = pd.read_csv("results.csv")  # Reload data to include gpd results
        # Add "diagonal" entries
        results["target"] = results["data"]
        results_cross = results_cross.append(results)

        for model in results["model"].unique():
            model_results(results_cross, model)

    if resampled:
        print("Generating resampled tables")
        results = pd.read_csv("results_resampled.csv")
        resampled_tables(results[results["target"] == "geofon"], suffix="_geofon")
        resampled_tables(results[results["target"] == "neic"], suffix="_neic")

        print("Generating resampled plots")
        resampled_plots(results[results["target"] == "geofon"], suffix="_geofon")
        resampled_plots(results[results["target"] == "neic"], suffix="_neic")


def resampled_tables(results, suffix):
    table = results_to_table(
        results,
        ["test_det_precision", "test_det_recall", "test_det_f1"],
        "dev_det_f1",
        ["P", "R", "F1"],
    )
    with open(f"results/resampled/detection_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_phase_precision", "test_phase_recall", "test_phase_f1"],
        "dev_phase_f1",
        ["P", "R", "F1"],
    )
    with open(f"results/resampled/phase_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_P_mean_s", "test_P_std_s", "test_P_mae_s"],
        "dev_P_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
    )
    with open(f"results/resampled/precision_p_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_S_mean_s", "test_S_std_s", "test_S_mae_s"],
        "dev_S_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
    )
    with open(f"results/resampled/precision_s_test{suffix}.tex", "w") as f:
        f.write(table)


def resampled_plots(results, suffix):
    fig = residual_matrix(
        "P", results, Path("pred_cross_resampled"), "dev_P_std_s", adjust_scale=False
    )
    fig.savefig(f"results/resampled/test_P_diff{suffix}.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix(
        "S", results, Path("pred_cross_resampled"), "dev_S_std_s", adjust_scale=False
    )
    fig.savefig(f"results/resampled/test_S_diff{suffix}.eps", bbox_inches="tight")
    plt.close(fig)


def model_results(results_cross, model):
    print(f"Generating {model} results")
    results_model = results_cross[results_cross["model"] == model]

    table = results_to_table(
        results_model,
        ["test_det_precision", "test_det_recall", "test_det_f1"],
        "dev_det_f1",
        ["P", "R", "F1"],
        axis=("data", "target"),
    )
    with open(f"results/cross/{model}_detection_test.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results_model[results_model["data"] != "lendb"],
        ["test_phase_precision", "test_phase_recall", "test_phase_f1"],
        "dev_phase_f1",
        ["P", "R", "F1"],
        axis=("data", "target"),
    )
    with open(f"results/cross/{model}_phase_test.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results_model[results_model["data"] != "lendb"],
        ["test_P_mean_s", "test_P_std_s", "test_P_mae_s"],
        "dev_P_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
        axis=("data", "target"),
    )
    with open(f"results/cross/{model}_precision_p_test.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results_model[results_model["data"] != "lendb"],
        ["test_S_mean_s", "test_S_std_s", "test_S_mae_s"],
        "dev_S_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
        axis=("data", "target"),
    )
    with open(f"results/cross/{model}_precision_s_test.tex", "w") as f:
        f.write(table)

    fig = residual_matrix(
        "P",
        results_model[results_model["data"] != "lendb"],
        [Path("pred"), Path("pred_cross")],
        "dev_P_std_s",
        axis=("data", "target"),
        separation=(4, 4),
    )
    fig.savefig(f"results/cross/{model}_test_P_diff.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix(
        "S",
        results_model[results_model["data"] != "lendb"],
        [Path("pred"), Path("pred_cross")],
        "dev_S_std_s",
        axis=("data", "target"),
        separation=(4, 4),
    )
    fig.savefig(f"results/cross/{model}_test_S_diff.eps", bbox_inches="tight")
    plt.close(fig)


def results_tables(results, suffix=None):
    if suffix is None:
        suffix = ""

    table = results_to_table(
        results,
        ["test_det_precision", "test_det_recall", "test_det_f1"],
        "dev_det_f1",
        ["P", "R", "F1"],
    )
    with open(f"results/detection_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_phase_precision", "test_phase_recall", "test_phase_f1"],
        "dev_phase_f1",
        ["P", "R", "F1"],
    )
    with open(f"results/phase_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_P_mean_s", "test_P_std_s", "test_P_mae_s"],
        "dev_P_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
    )
    with open(f"results/precision_p_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_S_mean_s", "test_S_std_s", "test_S_mae_s"],
        "dev_S_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
    )
    with open(f"results/precision_s_test{suffix}.tex", "w") as f:
        f.write(table)


def results_plots(results):
    fig = residual_matrix(
        "P", results, Path("pred"), "dev_P_std_s", separation=(4, None)
    )
    fig.savefig("results/test_P_diff.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix(
        "S", results, Path("pred"), "dev_S_std_s", separation=(4, None)
    )
    fig.savefig("results/test_S_diff.eps", bbox_inches="tight")
    plt.close(fig)


def detect_missing_entries(results):
    """
    Detects missing entries from the predictions grid assuming results should exist
    for all combinations of data, model and learning rate

    :param results: Results dataframe
    """
    for data in results["data"].unique():
        for model in results["model"].unique():
            for lr in results["lr"].unique():
                mask = np.logical_and(
                    results["data"] == data, results["model"] == model
                )
                mask = np.logical_and(mask, results["lr"] == lr)
                if np.sum(mask) != 1:
                    print(np.sum(mask), data, model, lr)


def results_to_array(results, cols, selection, minimize=False, axis=("data", "model")):
    """
    Packs results into numpy array with axis data, model, cols.

    :param results: Results dataframe
    :param cols: Columns to select from the dataframe
    :param selection: Column to use for selecting the best model configuration
                      (in case of multiple learning rates, ...)
    :param minimize: If true, chooses the minimal value from column selection, otherwise the maximal.
    :param axis: First two axis of the array, by default "data" and "model"
    :return: numpy array with results
    """
    ax0, ax1 = axis

    def generate_dict(ax):
        if ax == "data" or ax == "target":
            # Sort the datasets first by their class, than alphabetically
            data_class = [(DATA_CLASSES[data], data) for data in results[ax].unique()]
            data_class = sorted(data_class)
            keys = [x[1] for x in data_class]
        else:
            # Sort the models alphabetically
            keys = sorted(results[ax].unique())
        ax_dict = {x: i for i, x in enumerate(keys)}
        return ax_dict

    data_dict = generate_dict(ax0)
    model_dict = generate_dict(ax1)

    n_data = len(results[ax0].unique())
    n_model = len(results[ax1].unique())

    res_array = np.nan * np.zeros((n_data, n_model, len(cols)))

    for _, subdf in results.groupby([ax0, ax1]):
        if np.isnan(subdf[selection]).all():
            continue
        if minimize:
            i = np.nanargmin(subdf[selection])
        else:
            i = np.nanargmax(subdf[selection])
        row = subdf.iloc[i]
        entry = res_array[data_dict[row[ax0]], model_dict[row[ax1]]]
        for col_idx, col in enumerate(cols):
            entry[col_idx] = row[col]

    return data_dict, model_dict, res_array


def results_to_table(
    results,
    cols,
    selection,
    labels=None,
    minimize=False,
    average=None,
    axis=("data", "model"),
):
    """
    Format results as latex table

    :param results: Results dataframe
    :param cols: Columns to select from the dataframe
    :param selection: Column to use for selecting the best model configuration
                      (in case of multiple learning rates, ...)
    :param labels: Lables for the columns. Will be omitted if empty.
    :param minimize: If true, chooses the minimal value from column selection, otherwise the maximal.
    :param average: Indices for the columns that should have averages attached.
                    If None, outputs averages for all columns.
    :param axis: Axis of the table, by default "data" and "model"
    :return: String for table
    """
    ax0, ax1 = axis
    data_dict, model_dict, res_array = results_to_array(
        results, cols, selection, minimize=minimize, axis=axis
    )

    if average is None:
        average = list(range(len(cols)))

    n_data = len(results[ax0].unique())
    n_model = len(results[ax1].unique())

    inv_data_dict = {v: k for k, v in data_dict.items()}
    inv_model_dict = {v: k for k, v in model_dict.items()}

    header = f"\\backslashbox{{{ax0.capitalize()}}}{{{ax1.capitalize()}}}"
    header_count = 0
    for j in range(n_model):
        if np.isnan(res_array[:, j]).all():
            # print(f"Skipping {inv_model_dict[j]}")
            continue
        col_name = inv_model_dict[j]
        if ax1 == "data" or ax1 == "target":
            col_name = DATA_ALIASES[col_name]
        else:
            col_name = MODEL_ALIASES[col_name]

        header += f" & \multicolumn{{{len(cols)}}}{{|c}}{{{col_name}}}"
        header_count += 1
    header += f" & \multicolumn{{{len(average)}}}{{|c}}{{$\\diameter$}} \\\\"

    avg_labels = [label for i, label in enumerate(labels) if i in average]

    if labels is not None:
        label_str = " & ".join([""] + header_count * labels + avg_labels) + "\\\\"
    else:
        label_str = ""

    colspec = "|".join(["c"] + header_count * ["c" * len(cols)] + ["c" * len(average)])
    tabular = f"\\begin{{tabular}}{{{colspec}}}"

    table_str = ["\\setlength{\\tabcolsep}{2pt}", tabular, header, label_str, "\\hline"]

    for i in range(n_data):
        line = inv_data_dict[i]
        if ax0 == "data" or ax0 == "target":
            line = DATA_ALIASES[line]
        else:
            line = MODEL_ALIASES[line]

        if np.isnan(res_array[i]).all():
            # print(f"Skipping {inv_data_dict[i]}")
            continue
        for j in range(n_model):
            if np.isnan(res_array[:, j]).all():
                # print(f"Skipping {inv_model_dict[j]}")
                continue
            for col_idx, _ in enumerate(cols):
                line += f" & {res_array[i, j, col_idx]:.2f}"

        avg = np.nanmean(res_array[i], axis=0)
        for col_idx, _ in enumerate(cols):
            if col_idx in average:
                line += f" & {avg[col_idx]:.2f}"

        line += " \\\\"
        table_str.append(line)

    table_str.append("\\hline")

    line = "$\\diameter$"
    for j in range(n_model):
        if np.isnan(res_array[:, j]).all():
            # print(f"Skipping {inv_model_dict[j]}")
            continue
        avg = np.nanmean(res_array[:, j], axis=0)

        for col_idx, _ in enumerate(cols):
            if col_idx in average:
                line += f" & {avg[col_idx]:.2f}"
            else:
                line += " &"

    line += " \\\\"
    table_str.append(line)
    table_str.append("\\end{tabular}")

    return "\n".join(table_str)


def residual_matrix(
    phase,
    results,
    pred_path,
    selection,
    lim=1.5,
    axis=("data", "model"),
    separation=(None, None),
    adjust_scale=True,
):
    """
    Plots pick residual distributions for each model and dataset in a grid

    :param phase: Seismic phase to plot
    :param results: Results dataframe
    :param pred_path: Path to predictions or list of path to predictions
    :param selection: Column to use for selecting the best model configuration
                      (in case of multiple learning rates, ...)
    :param lim: Plotting limit for the distributions
    :param axis: Axis of the grid, by default "data" and "model"
    :param separation: Points for splitting grid. If non, won't split grid in this axis.
    :return: Matplotlib figure handle
    """
    if not isinstance(pred_path, list):
        pred_path = [pred_path]

    ax0, ax1 = axis
    sep0, sep1 = separation
    data_dict, model_dict, res_array = results_to_array(
        results, [f"test_{phase}_mean_s"], selection, minimize=True, axis=axis
    )
    res_array = res_array[:, :, 0]

    n_data = len(results[ax0].unique())
    n_model = len(results[ax1].unique())

    inv_data_dict = {v: k for k, v in data_dict.items()}
    inv_model_dict = {v: k for k, v in model_dict.items()}

    true_n_data = np.sum((~np.isnan(res_array)).any(axis=1))
    true_n_model = np.sum((~np.isnan(res_array)).any(axis=0))

    if true_n_data == 0 or true_n_model == 0:
        return plt.figure()

    fig = plt.figure(figsize=(15, 15))

    def splits_from_sep(sep, true_n):
        if sep is None:
            splits = (true_n,)
        else:
            splits = (sep, true_n - sep)
        return splits

    ax0splits = splits_from_sep(sep0, true_n_data)
    ax1splits = splits_from_sep(sep1, true_n_model)

    gs = fig.add_gridspec(
        len(ax0splits),
        len(ax1splits),
        hspace=0.1,
        wspace=0.1,
        height_ratios=ax0splits,
        width_ratios=ax1splits,
    )
    sub_gslist = []
    for i in range(len(ax0splits)):
        tmp = []
        for j in range(len(ax1splits)):
            tmp.append(
                gs[i, j].subgridspec(
                    ax0splits[i], ax1splits[j], hspace=0.05, wspace=0.05
                )
            )
        sub_gslist.append(tmp)

    if sep0 is None:
        sep0 = true_n_data
    if sep1 is None:
        sep1 = true_n_model

    axs = np.empty((true_n_data, true_n_model), dtype=object)
    for i in range(true_n_data):
        for j in range(true_n_model):
            if i < sep0:
                if j < sep1:
                    axs[i, j] = fig.add_subplot(sub_gslist[0][0][i, j])
                else:
                    axs[i, j] = fig.add_subplot(sub_gslist[0][1][i, j - sep1])
            else:
                if j < sep1:
                    axs[i, j] = fig.add_subplot(sub_gslist[1][0][i - sep0, j])
                else:
                    axs[i, j] = fig.add_subplot(sub_gslist[1][1][i - sep0, j - sep1])

    true_i = 0
    for i in range(n_data):
        if np.isnan(res_array[i]).all():
            continue
        true_j = 0
        for j in range(n_model):
            local_lim = lim
            if true_i < sep0 and true_j < sep1 and adjust_scale:
                local_lim = 0.45
                axs[true_i, true_j].set_xticks([-0.3, 0, 0.3])

            if np.isnan(res_array[:, j]).all():
                continue
            data, model = inv_data_dict[i], inv_model_dict[j]
            mask = np.logical_and(results[ax1] == model, results[ax0] == data)

            axs[true_i, true_j].set_yticks([])
            axs[true_i, true_j].set_yticklabels([])
            axs[true_i, true_j].set_xlim(-local_lim, local_lim)

            subdf = results[mask]
            if np.isnan(subdf[selection]).all():
                true_j += 1
                continue
            lr_idx = np.nanargmin(subdf[selection])
            row = subdf.iloc[lr_idx]

            for pred_path_member in pred_path:
                pred_path_loc = (
                    pred_path_member
                    / row["experiment"]
                    / f"version_{row['version']}"
                    / "test_task23.csv"
                )
                if pred_path_loc.is_file():
                    break
            if not pred_path_loc.is_file():
                true_j += 1
                continue
            pred = pd.read_csv(pred_path_loc)

            pred_phase = pred[pred["phase_label"] == phase]
            pred_col = f"{phase.lower()}_sample_pred"

            diff = (pred_phase[pred_col] - pred_phase["phase_onset"]) / pred_phase[
                "sampling_rate"
            ]

            axs[true_i, true_j].axvline(np.mean(diff), c="C1", lw=3)
            axs[true_i, true_j].axvline(np.median(diff), c="C2", lw=3)

            bins = np.linspace(-local_lim, local_lim, 50)
            axs[true_i, true_j].hist(diff, bins=bins)

            if ax1 == "model":
                title = MODEL_ALIASES[model]
            else:
                title = DATA_ALIASES[model]
            axs[0, true_j].set_title(title)

            outliers = np.abs(diff) > local_lim
            frac_outliers = np.sum(outliers) / len(diff)
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(np.mean(diff ** 2))

            lineheight = 0.13
            axs[true_i, true_j].text(
                0.98,
                0.98,
                f"{frac_outliers:.2f}",
                transform=axs[true_i, true_j].transAxes,
                ha="right",
                va="top",
            )
            axs[true_i, true_j].text(
                0.98,
                0.98 - lineheight,
                f"{mae:.2f}",
                transform=axs[true_i, true_j].transAxes,
                ha="right",
                va="top",
            )
            axs[true_i, true_j].text(
                0.98,
                0.98 - 2 * lineheight,
                f"{rmse:.2f}",
                transform=axs[true_i, true_j].transAxes,
                ha="right",
                va="top",
            )
            axs[true_i, true_j].text(
                0.02,
                0.98,
                "OUT",
                transform=axs[true_i, true_j].transAxes,
                ha="left",
                va="top",
            )
            axs[true_i, true_j].text(
                0.02,
                0.98 - lineheight,
                "MAE",
                transform=axs[true_i, true_j].transAxes,
                ha="left",
                va="top",
            )
            axs[true_i, true_j].text(
                0.02,
                0.98 - 2 * lineheight,
                "RMSE",
                transform=axs[true_i, true_j].transAxes,
                ha="left",
                va="top",
            )

            if data == "neic" and model == "eqtransformer":
                axs[true_i, true_j].set_facecolor("#aaaaaa")

            true_j += 1

        if ax0 == "model":
            ylabel = MODEL_ALIASES[data]
        else:
            ylabel = DATA_ALIASES[data]
        axs[true_i, 0].set_ylabel(ylabel)

        true_i += 1

    for ax in axs[-1]:
        ax.set_xlabel("$t_{pred} - t_{true}$")

    mid_left_ax = axs[axs.shape[0] // 2, 0]
    mid_left_ax.set_ylabel(ax0.capitalize() + "\n" + mid_left_ax.get_ylabel())
    mid_top_ax = axs[0, axs.shape[1] // 2]
    mid_top_ax.set_title(ax1.capitalize() + "\n" + mid_top_ax.get_title())

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots and tables for results"
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="If true, creates outputs for basic experiments (including GPD).",
    )
    parser.add_argument(
        "--cross",
        action="store_true",
        help="If true, creates outputs for cross-domain experiments.",
    )
    parser.add_argument(
        "--resampled",
        action="store_true",
        help="If true, creates outputs for resampled experiments.",
    )

    args = parser.parse_args()
    main(args.base, args.cross, args.resampled)
