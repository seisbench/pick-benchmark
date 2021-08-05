import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(font_scale=1.5)
sns.set_style("ticks")
sns.set_palette("colorblind")


def main():
    results = pd.read_csv("results.csv")

    detect_missing_entries(results)

    print("Generating tables")
    results_tables(results)

    print("Generating plots")
    results_plots(results)

    results_cross = pd.read_csv("results_cross.csv")
    # Add "diagonal" entries
    results["target"] = results["data"]
    results_cross = results_cross.append(results)

    for model in results["model"].unique():
        model_results(results_cross, model)


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
    )
    fig.savefig(f"results/cross/{model}_test_P_diff.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix(
        "S",
        results_model[results_model["data"] != "lendb"],
        [Path("pred"), Path("pred_cross")],
        "dev_S_std_s",
        axis=("data", "target"),
    )
    fig.savefig(f"results/cross/{model}_test_S_diff.eps", bbox_inches="tight")
    plt.close(fig)


def results_tables(results):
    table = results_to_table(
        results,
        ["test_det_precision", "test_det_recall", "test_det_f1"],
        "dev_det_f1",
        ["P", "R", "F1"],
    )
    with open("results/detection_test.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_phase_precision", "test_phase_recall", "test_phase_f1"],
        "dev_phase_f1",
        ["P", "R", "F1"],
    )
    with open("results/phase_test.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_P_mean_s", "test_P_std_s", "test_P_mae_s"],
        "dev_P_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
    )
    with open("results/precision_p_test.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_S_mean_s", "test_S_std_s", "test_S_mae_s"],
        "dev_S_std_s",
        ["$\\mu$", "$\\sigma$", "MAE"],
        minimize=True,
        average=[1, 2],
    )
    with open("results/precision_s_test.tex", "w") as f:
        f.write(table)


def results_plots(results):
    fig = residual_matrix("P", results, Path("pred"), "dev_P_std_s")
    fig.savefig("results/test_P_diff.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix("S", results, Path("pred"), "dev_S_std_s")
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

    data_dict = {data: i for i, data in enumerate(sorted(results[ax0].unique()))}
    model_dict = {model: i for i, model in enumerate(sorted(results[ax1].unique()))}

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

    header = f"\\backslashbox{{{ax0}}}{{{ax1}}}"
    header_count = 0
    for j in range(n_model):
        if np.isnan(res_array[:, j]).all():
            # print(f"Skipping {inv_model_dict[j]}")
            continue
        header += f" & \multicolumn{{{len(cols)}}}{{|c}}{{{inv_model_dict[j]}}}"
        header_count += 1
    header += f" & \multicolumn{{{len(average)}}}{{|c}}{{average}} \\\\"

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

    line = "average"
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
    phase, results, pred_path, selection, lim=2, axis=("data", "model")
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
    :return: Matplotlib figure handle
    """
    if not isinstance(pred_path, list):
        pred_path = [pred_path]

    ax0, ax1 = axis
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
    axs = fig.subplots(
        true_n_data,
        true_n_model,
        sharex=True,
        sharey=False,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    true_i = 0
    for i in range(n_data):
        if np.isnan(res_array[i]).all():
            continue
        true_j = 0
        for j in range(n_model):
            if np.isnan(res_array[:, j]).all():
                continue
            data, model = inv_data_dict[i], inv_model_dict[j]
            mask = np.logical_and(results[ax1] == model, results[ax0] == data)

            axs[true_i, true_j].set_yticklabels([])
            axs[true_i, true_j].set_xlim(-lim, lim)

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

            bins = np.linspace(-lim, lim, 50)
            axs[true_i, true_j].hist(diff, bins=bins)

            axs[0, true_j].set_title(model)

            outliers = np.abs(diff) > lim
            frac_outliers = np.sum(outliers) / len(diff)
            diff[outliers] = lim
            # std_outlierfree = np.std(diff)
            axs[true_i, true_j].text(
                0.98,
                0.98,
                f"{frac_outliers:.2f}",
                transform=axs[true_i, true_j].transAxes,
                ha="right",
                va="top",
            )
            # axs[true_i, true_j].text(0.98, 0.87, f"{std_outlierfree:.2f}", transform=axs[true_i, true_j].transAxes, ha="right", va="top")

            true_j += 1

        axs[true_i, 0].set_ylabel(data)

        true_i += 1

    for ax in axs[-1]:
        ax.set_xlabel("$t_{pred} - t_{true}$")

    mid_left_ax = axs[axs.shape[0] // 2, 0]
    mid_left_ax.set_ylabel(ax0 + "\n" + mid_left_ax.get_ylabel())
    mid_top_ax = axs[0, axs.shape[1] // 2]
    mid_top_ax.set_title(ax1 + "\n" + mid_top_ax.get_title())

    return fig


if __name__ == "__main__":
    main()
