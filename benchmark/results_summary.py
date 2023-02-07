"""
This file contains functionality to generate results tables and plots.
Please note that this script is adapted specifically to the benchmark paper.
Therefore, manual adjustments will be required when applying it in other scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpl
import seaborn as sns
from pathlib import Path
import logging
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
import seisbench.data as sbd

sns.set(font_scale=1.5)
sns.set_style("ticks")
sns.set_palette("colorblind")


# Maps internal model names to model names in plots and tables
MODEL_ALIASES = {
    "baer": "Baer-Kradolfer",
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
# Maps internal model names to model abbreviations in plots and tables
MODEL_ABBREVIATIONS = {
    "basicphaseae": "BPAE",
    "cred": "CRED",
    "dppdetect": "DPP",
    "dpppickerp": "DPP",
    "dpppickers": "DPP",
    "eqtransformer": "EQT",
    "gpd": "GPD-O",
    "gpdpick": "GPD",
    "phasenet": "PN",
}
# Maps internal data names to model names in plots and tables
DATA_ALIASES = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "instance": "INSTANCE",
    "iquique": "Iquique",
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
    "iquique": "regional",
    "lendb": "regional",
    "neic": "tele",
    "scedc": "regional",
    "stead": "regional",
}
# Provides limits for the ROC curves
ROC_LIMITS = {
    "ethz": 0.3,
    "geofon": 0.5,
    "instance": 0.3,
    "iquique": 0.3,
    "lendb": 0.3,
    "scedc": 0.5,
    "stead": 0.1,
}
MODEL_COLORS = {
    "baer": "C6",
    "basicphaseae": "C0",
    "cred": "C1",
    "dppdetect": "C2",
    "dpppickerp": "C2",
    "dpppickers": "C2",
    "eqtransformer": "C3",
    "gpd": "C7",
    "gpdpick": "C4",
    "phasenet": "C5",
}


def main(base, cross, resampled, roc, roc_cross, phase_cross, thresholds, snr):
    if not (
        base
        or cross
        or resampled
        or roc
        or roc_cross
        or phase_cross
        or thresholds
        or snr
    ):
        logging.warning("No task selected. exiting.")

    if base:
        results = pd.read_csv("results.csv")
        results = results[results["model"] != "phasenetlight"]

        detect_missing_entries(results)
        print("Generating tables")
        results_tables(results, suffix="_gpd")
        results = results[results["model"] != "gpd"]
        results_tables(results)

        results_baer = pd.read_csv("results_baer.csv")
        results_baer = results_baer[
            results_baer["data"] == results_baer["target"]
        ]  # Only consider in-domain results
        results = results.append(results_baer)

        print("Generating plots")
        results_plots(results)

    if roc:
        results = pd.read_csv("results.csv")
        results = results[results["model"] != "phasenetlight"]
        results = results[results["model"] != "gpd"]

        print("Generating ROC")
        fig = results_roc(results, "dev_det_auc")
        fig.savefig("results/detection_roc.eps", bbox_inches="tight")

        fig = results_roc(results, "dev_det_auc", double_axis=True, cols=3)
        fig.savefig("results/detection_roc_double.eps", bbox_inches="tight")

        fig = results_roc(results, "dev_det_auc", cols=3)
        fig.savefig("results/detection_roc_transposed.eps", bbox_inches="tight")

        fig = results_roc(results, "dev_det_auc", full_axis=True)
        fig.savefig("results/detection_roc_full.eps", bbox_inches="tight")

    if cross:
        results_cross = pd.read_csv("results_cross.csv")
        results_cross = results_cross[results_cross["model"] != "phasenetlight"]
        results = pd.read_csv("results.csv")  # Reload data to include gpd results
        results = results[results["model"] != "phasenetlight"]
        # Add "diagonal" entries
        results["target"] = results["data"]
        results_cross = results_cross.append(results)
        results_baer = pd.read_csv("results_baer.csv")
        results_cross = results_cross.append(results_baer)

        for model in results_cross["model"].unique():
            model_results(results_cross, model)

    if roc_cross:
        results_cross = pd.read_csv("results_cross.csv")
        results_cross = results_cross[results_cross["model"] != "phasenetlight"]
        results = pd.read_csv("results.csv")
        results = results[results["model"] != "phasenetlight"]
        # Add "diagonal" entries
        results["target"] = results["data"]
        results_cross = results_cross.append(results)

        fig = results_auc_cross(results_cross, "dev_det_auc")
        fig.savefig("results/detection_auc_cross.eps", bbox_inches="tight")

        fig = results_roc_cross(results_cross, "dev_det_auc")
        fig.savefig("results/detection_roc_cross.eps", bbox_inches="tight")

    if phase_cross:
        results_cross = pd.read_csv("results_cross.csv")
        results_cross = results_cross[results_cross["model"] != "phasenetlight"]
        results = pd.read_csv("results.csv")
        results = results[results["model"] != "phasenetlight"]
        # Add "diagonal" entries
        results["target"] = results["data"]
        results_cross = results_cross.append(results)

        fig = results_phase_cross(results_cross, "S")
        fig.savefig("results/S_diff_cross.eps", bbox_inches="tight")

        results_baer = pd.read_csv("results_baer.csv")
        results_cross = results_cross.append(results_baer)
        fig = results_phase_cross(results_cross, "P")
        fig.savefig("results/P_diff_cross.eps", bbox_inches="tight")

    if resampled:
        print("Generating resampled tables")
        results = pd.read_csv("results_resampled.csv")
        results = results[results["model"] != "phasenetlight"]
        resampled_tables(results[results["target"] == "geofon"], suffix="_geofon")
        resampled_tables(results[results["target"] == "neic"], suffix="_neic")

        print("Generating resampled plots")
        resampled_plots(results[results["target"] == "geofon"], suffix="_geofon")
        resampled_plots(results[results["target"] == "neic"], suffix="_neic")

    if thresholds:
        results = pd.read_csv("results.csv")
        results = results[results["model"] != "phasenetlight"]

        table = results_to_table(
            results,
            ["det_threshold"],
            "dev_det_auc",
            ["Thr"],
        )
        with open(f"results/detection_thresholds.tex", "w") as f:
            f.write(table)

        table = results_to_table(
            results,
            ["phase_threshold"],
            "dev_phase_mcc",
            ["Thr"],
        )
        with open(f"results/phase_thresholds.tex", "w") as f:
            f.write(table)

    if snr:
        results = pd.read_csv("results.csv")
        results = results[results["model"] != "phasenetlight"]
        results = results[results["model"] != "gpd"]

        results_baer = pd.read_csv("results_baer.csv")
        results_baer = results_baer[
            results_baer["data"] == results_baer["target"]
        ]  # Only consider in-domain results
        results = results.append(results_baer)

        fig = stead_snr(results, "dev_P_std_s")
        fig.savefig("results/stead_snr.eps", bbox_inches="tight")


def resampled_tables(results, suffix):
    table = results_to_table(
        results,
        ["test_det_auc"],
        "dev_det_auc",
        ["AUC"],
    )
    with open(f"results/resampled/detection_test{suffix}.tex", "w") as f:
        f.write(table)

    table = results_to_table(
        results,
        ["test_phase_mcc"],
        "dev_phase_mcc",
        ["MCC"],
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
        ["test_phase_mcc"],
        "dev_phase_mcc",
        ["MCC"],
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
        [Path("pred"), Path("pred_cross"), Path("pred_baer")],
        "dev_P_std_s",
        axis=("data", "target"),
        separation=(5, 5),
    )
    fig.savefig(f"results/cross/{model}_test_P_diff.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix(
        "S",
        results_model[results_model["data"] != "lendb"],
        [Path("pred"), Path("pred_cross")],
        "dev_S_std_s",
        axis=("data", "target"),
        separation=(5, 5),
    )
    fig.savefig(f"results/cross/{model}_test_S_diff.eps", bbox_inches="tight")
    plt.close(fig)


def results_tables(results, suffix=None):
    if suffix is None:
        suffix = ""

    table = results_to_table(
        results,
        ["test_det_auc"],
        "dev_det_auc",
        ["AUC"],
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
        ["test_phase_mcc"],
        "dev_phase_mcc",
        ["MCC"],
    )
    with open(f"results/phase_test_mcc{suffix}.tex", "w") as f:
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
        "P",
        results,
        [Path("pred"), Path("pred_baer")],
        "dev_P_std_s",
        separation=(5, None),
    )
    fig.savefig("results/test_P_diff.eps", bbox_inches="tight")
    plt.close(fig)

    fig = residual_matrix(
        "S", results, Path("pred"), "dev_S_std_s", separation=(5, None)
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


def results_auc_cross(results, selection):
    results = results[
        ~np.isnan(results["dev_det_f1"])
    ]  # Filter out invalid model/data/target combinations
    results = results[results["model"] != "gpd"]  # Remove original GPD variant
    model_list = sorted(results["model"].unique())

    model_res = {}
    data_dict = None
    target_dict = None
    for model in model_list:
        tmp_data_dict, tmp_target_dict, res_array = results_to_array(
            results[results["model"] == "phasenet"],
            ["test_det_auc"],
            selection,
            minimize=False,
            axis=("data", "target"),
        )
        model_res[model] = res_array[:, :, 0]

        # Ensure dicts are compatible
        if data_dict is None:
            data_dict = tmp_data_dict
            target_dict = tmp_target_dict
        assert data_dict == tmp_data_dict
        assert target_dict == tmp_target_dict

    n_data = len(data_dict)
    n_target = len(target_dict)

    inv_data_dict = {v: k for k, v in data_dict.items()}
    inv_target_dict = {v: k for k, v in target_dict.items()}

    res_array = model_res["phasenet"]  # Dummy for removing invalid datasets
    true_n_data = np.sum((~np.isnan(res_array)).any(axis=1))
    true_n_target = np.sum((~np.isnan(res_array)).any(axis=0))

    if true_n_data == 0 or true_n_target == 0:
        return plt.figure()

    fig = plt.figure(figsize=(3 * true_n_target, 3 * true_n_data))
    axs = fig.subplots(
        true_n_data, true_n_target, gridspec_kw={"hspace": 0.075, "wspace": 0.075}
    )

    true_i = 0
    for i in range(n_data):
        if np.isnan(res_array[i]).all():
            continue

        data = inv_data_dict[i]
        axs[true_i, 0].set_ylabel(DATA_ALIASES[data] + "\nAUC")

        true_j = 0
        for j in range(n_target):
            ax = axs[true_i, true_j]
            if np.isnan(res_array[:, j]).all():
                continue
            data, target = inv_data_dict[i], inv_target_dict[j]

            if true_i == 0:
                ax.set_title(DATA_ALIASES[target])

            for model_idx, model in enumerate(model_list):
                mask = np.logical_and(
                    results["target"] == target, results["data"] == data
                )
                mask = np.logical_and(mask, results["model"] == model)
                subdf = results[mask]
                if np.isnan(subdf[selection]).all():
                    continue
                lr_idx = np.nanargmax(subdf[selection])
                row = subdf.iloc[lr_idx]

                ax.bar(model_idx, row["test_det_auc"])

            ax.set_ylim(0.5, 1)

            if true_j != 0:
                ax.set_yticklabels([])
            if true_i != true_n_data - 1:
                ax.set_xticklabels([])

            true_j += 1

        true_i += 1

    for ax in axs[-1]:
        ax.set_xticks(np.arange(len(model_list)))
        ax.set_xticklabels([MODEL_ALIASES[x] for x in model_list], rotation=90)

    mid_left_ax = axs[axs.shape[0] // 2, 0]
    mid_left_ax.set_ylabel("Data\n" + mid_left_ax.get_ylabel())
    mid_top_ax = axs[0, axs.shape[1] // 2]
    mid_top_ax.set_title("Target\n" + mid_top_ax.get_title())

    return fig


def results_phase_cross(results, phase):
    selection = f"dev_{phase}_std_s"
    pred_path = [Path("pred"), Path("pred_cross")]
    if phase == "P":
        pred_path.append(Path("pred_baer"))

    results = results[
        ~np.isnan(results[selection])
    ]  # Filter out invalid model/data/target combinations
    results = results[results["model"] != "gpd"]  # Remove original GPD variant
    # Remove incorrect DPP variants
    results = results[results["model"] != "dppdetect"]
    if phase == "S":
        results = results[results["model"] != "dpppickerp"]
    elif phase == "S":
        results = results[results["model"] != "dpppickers"]
    model_list = sorted(results["model"].unique())

    # Dummy for removing invalid datasets
    data_dict, target_dict, res_array = results_to_array(
        results[results["model"] == "phasenet"],
        ["test_P_mean_s"],
        selection,
        minimize=True,
        axis=("data", "target"),
    )
    res_array = res_array[:, :, 0]

    n_data = len(data_dict)
    n_target = len(target_dict)

    inv_data_dict = {v: k for k, v in data_dict.items()}
    inv_target_dict = {v: k for k, v in target_dict.items()}

    true_n_data = np.sum((~np.isnan(res_array)).any(axis=1))
    true_n_target = np.sum((~np.isnan(res_array)).any(axis=0))

    if true_n_data == 0 or true_n_target == 0:
        return plt.figure()

    fig = plt.figure(figsize=(3 * true_n_target, 3 * true_n_data))
    # axs = fig.subplots(
    #    true_n_data, true_n_target, gridspec_kw={"hspace": 0.075, "wspace": 0.075}
    # )
    gs = fig.add_gridspec(1, 2, width_ratios=(true_n_target - 2, 2), wspace=0.1)
    gs1 = gs[0].subgridspec(true_n_data, true_n_target - 2, hspace=0.075, wspace=0.075)
    gs2 = gs[1].subgridspec(true_n_data, 2, hspace=0.075, wspace=0.075)
    axs = np.empty((true_n_data, true_n_target), dtype=object)
    for i in range(true_n_data):
        for j in range(true_n_target):
            if j < true_n_target - 2:
                axs[i, j] = fig.add_subplot(gs1[i, j])
            else:
                axs[i, j] = fig.add_subplot(gs2[i, j - true_n_target + 2])

    true_i = 0
    for i in range(n_data):
        if np.isnan(res_array[i]).all():
            continue

        data = inv_data_dict[i]
        axs[true_i, 0].set_ylabel(DATA_ALIASES[data] + "\n$t_{pred} - t_{true}~[s]$")

        true_j = 0
        for j in range(n_target):
            ax = axs[true_i, true_j]
            if np.isnan(res_array[:, j]).all():
                continue
            data, target = inv_data_dict[i], inv_target_dict[j]
            print(data, target)

            if true_i == 0:
                ax.set_title(DATA_ALIASES[target])

            for model_idx, model in enumerate(model_list):
                mask = np.logical_and(
                    results["target"] == target, results["data"] == data
                )
                mask = np.logical_and(mask, results["model"] == model)
                subdf = results[mask]
                if np.isnan(subdf[selection]).all():
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

                    # For Bear picker, needs line without version
                    pred_path_loc = (
                        pred_path_member / row["experiment"] / "test_task23.csv"
                    )
                    if pred_path_loc.is_file():
                        break

                if not pred_path_loc.is_file():
                    continue

                pred = pd.read_csv(pred_path_loc)
                pred = pred[pred["phase_label"] == phase]

                diff = (
                    pred[f"{phase.lower()}_sample_pred"] - pred["phase_onset"]
                ) / pred["sampling_rate"]
                boxtop = np.quantile(diff, 0.75)
                boxbot = np.quantile(diff, 0.25)
                whisktop = np.quantile(diff, 0.9)
                whiskbot = np.quantile(diff, 0.1)

                def plot_whisk(b, t):
                    ax.plot(
                        [model_idx - 0.25, model_idx + 0.25],
                        [b, b],
                        "k-",
                        solid_capstyle="butt",
                    )
                    ax.plot(
                        [model_idx - 0.25, model_idx + 0.25],
                        [t, t],
                        "k-",
                        solid_capstyle="butt",
                    )
                    ax.plot([model_idx, model_idx], [b, t], "k-", solid_capstyle="butt")

                plot_whisk(whiskbot, whisktop)
                ax.bar(
                    model_idx,
                    height=boxtop - boxbot,
                    bottom=boxbot,
                    color=MODEL_COLORS[model],
                )
                ax.plot(
                    [model_idx - 0.4, model_idx + 0.4],
                    [np.mean(diff), np.mean(diff)],
                    linestyle=":",
                    color="k",
                    lw=3,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [model_idx - 0.4, model_idx + 0.4],
                    [np.median(diff), np.median(diff)],
                    linestyle="-",
                    color="k",
                    lw=3,
                    solid_capstyle="butt",
                )

            ax.set_xlim(-0.6, len(model_list) - 0.4)
            if target in ["neic", "geofon"]:
                ax.set_ylim(-0.75 * 3, 0.75 * 3)
            else:
                ax.set_ylim(-0.75, 0.75)

            if true_j != 0 and true_j != true_n_target - 2:
                ax.set_yticklabels([])
            if true_i != true_n_data - 1:
                ax.set_xticklabels([])

            true_j += 1

        true_i += 1

    for ax in axs[-1]:
        ax.set_xticks(np.arange(len(model_list)))
        ax.set_xticklabels([MODEL_ALIASES[x] for x in model_list], rotation=90)

    mid_left_ax = axs[axs.shape[0] // 2, 0]
    mid_left_ax.set_ylabel("Data\n" + mid_left_ax.get_ylabel())
    mid_top_ax = axs[0, axs.shape[1] // 2]
    mid_top_ax.set_title("Target\n" + mid_top_ax.get_title())

    return fig


def results_roc_cross(results, selection):
    results = results[
        ~np.isnan(results["dev_det_f1"])
    ]  # Filter out invalid model/data/target combinations
    results = results[results["model"] != "gpd"]  # Remove original GPD variant
    data_dict, target_dict, res_array = results_to_array(
        results[results["model"] == "phasenet"],
        ["test_det_f1"],
        selection,
        minimize=False,
        axis=("data", "target"),
    )
    model_list = sorted(results["model"].unique())
    pred_path = [Path("pred"), Path("pred_cross")]

    res_array = res_array[:, :, 0]

    n_data = len(data_dict)
    n_target = len(target_dict)

    inv_data_dict = {v: k for k, v in data_dict.items()}
    inv_target_dict = {v: k for k, v in target_dict.items()}

    true_n_data = np.sum((~np.isnan(res_array)).any(axis=1))
    true_n_target = np.sum((~np.isnan(res_array)).any(axis=0))

    if true_n_data == 0 or true_n_target == 0:
        return plt.figure()

    fig = plt.figure(figsize=(4 * true_n_target, 4 * true_n_data))
    gs = fig.add_gridspec(1, 2, width_ratios=(true_n_target - 1, 1), wspace=0.1)
    gs1 = gs[0].subgridspec(true_n_data, true_n_target - 1, hspace=0.05, wspace=0.05)
    gs2 = gs[1].subgridspec(true_n_data, 1, hspace=0.05)
    axs = np.empty((true_n_data, true_n_target), dtype=object)
    for i in range(true_n_data):
        for j in range(true_n_target):
            if j < true_n_target - 1:
                axs[i, j] = fig.add_subplot(gs1[i, j])
            else:
                axs[i, j] = fig.add_subplot(gs2[i])

    lineheight = 0.08

    true_i = 0
    for i in range(n_data):
        if np.isnan(res_array[i]).all():
            continue

        data = inv_data_dict[i]
        axs[true_i, 0].set_ylabel(DATA_ALIASES[data] + "\ntrue positive rate")

        true_j = 0
        for j in range(n_target):
            ax = axs[true_i, true_j]
            ax.set_aspect("equal")
            if np.isnan(res_array[:, j]).all():
                continue
            data, target = inv_data_dict[i], inv_target_dict[j]

            if true_i == 0:
                ax.set_title(DATA_ALIASES[target])

            for model_idx, model in enumerate(model_list):
                mask = np.logical_and(
                    results["target"] == target, results["data"] == data
                )
                mask = np.logical_and(mask, results["model"] == model)
                subdf = results[mask]
                if np.isnan(subdf[selection]).all():
                    continue
                lr_idx = np.nanargmax(subdf[selection])
                row = subdf.iloc[lr_idx]

                for pred_path_member in pred_path:
                    pred_path_loc = (
                        pred_path_member
                        / row["experiment"]
                        / f"version_{row['version']}"
                        / "test_task1.csv"
                    )
                    if pred_path_loc.is_file():
                        break
                if not pred_path_loc.is_file():
                    continue

                pred = pd.read_csv(pred_path_loc)
                pred["trace_type_bin"] = pred["trace_type"] == "earthquake"

                fpr, tpr, thr = roc_curve(
                    pred["trace_type_bin"], pred["score_detection"]
                )
                auc = roc_auc_score(pred["trace_type_bin"], pred["score_detection"])
                idx = np.argmin(
                    thr > row["det_threshold"]
                )  # Index of optimal threshold in terms of F1 score

                ax.text(
                    0.98,
                    0.01 + model_idx * lineheight,
                    f"{MODEL_ABBREVIATIONS[model]} {auc:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    color=f"C{model_idx}",
                    zorder=102,
                )

                ax.plot(fpr, tpr, label=model, color=f"C{model_idx}")
                ax.plot(fpr[idx], tpr[idx], "D", label=model, color=f"C{model_idx}")

            lim = 0.5
            if target == "geofon":
                lim = 1.02
            ax.set_xlim(-lim / 50, lim)
            ax.set_ylim(1 - lim, 1 + lim / 50)

            if true_j != 0 and true_j != true_n_target - 1:
                ax.set_yticklabels([])
            if true_i != true_n_data - 1:
                ax.set_xticklabels([])

            bg_text = mpl.Rectangle(
                (lim / 2.3, 1 - lim + 0.02 * lim),
                lim - lim / 2.3 - 0.02 * lim,
                lim / 2,
                zorder=101,
                linewidth=0,
                edgecolor=None,
                facecolor="#eeeeee",
            )
            ax.add_patch(bg_text)

            true_j += 1

        true_i += 1

    for ax in axs[-1]:
        ax.set_xlabel("false positive rate")

    mid_left_ax = axs[axs.shape[0] // 2, 0]
    mid_left_ax.set_ylabel("Data\n" + mid_left_ax.get_ylabel())
    mid_top_ax = axs[0, axs.shape[1] // 2]
    mid_top_ax.set_title("Target\n" + mid_top_ax.get_title())

    return fig


def results_roc(results, selection, cols=2, full_axis=False, double_axis=False):
    data_dict, model_dict, res_array = results_to_array(
        results,
        ["test_det_precision", "test_det_recall", "test_det_f1"],
        selection,
        minimize=False,
    )
    pred_path = [Path("pred")]

    res_array = res_array[:, :, 0]

    n_data = len(data_dict)
    n_model = len(model_dict)

    inv_data_dict = {v: k for k, v in data_dict.items()}
    inv_model_dict = {v: k for k, v in model_dict.items()}

    true_n_data = np.sum((~np.isnan(res_array)).any(axis=1))
    true_n_model = np.sum((~np.isnan(res_array)).any(axis=0))

    rows = int(np.ceil(true_n_data / cols))

    if true_n_data == 0 or true_n_model == 0:
        return plt.figure()

    panel_size = 5
    lineheight = 0.07
    row_spacing = 1  # Use every row
    if double_axis:
        # cols = true_n_data
        rows *= 2
        row_spacing = 2  # Use every second row

    fig = plt.figure(figsize=(panel_size * cols, panel_size * rows))
    axs = fig.subplots(rows, cols)

    true_i = 0
    for i in range(n_data):
        if np.isnan(res_array[i]).all():
            continue
        true_j = 0
        tmp_ax = [axs[true_i // cols * row_spacing, true_i % cols]]
        if double_axis:
            tmp_ax.append(axs[true_i // cols * row_spacing + 1, true_i % cols])
        for j in range(n_model):
            if np.isnan(res_array[:, j]).all():
                continue
            data, model = inv_data_dict[i], inv_model_dict[j]

            mask = np.logical_and(results["model"] == model, results["data"] == data)
            subdf = results[mask]
            if np.isnan(subdf[selection]).all():
                true_j += 1
                continue
            lr_idx = np.nanargmax(subdf[selection])
            row = subdf.iloc[lr_idx]

            for pred_path_member in pred_path:
                pred_path_loc = (
                    pred_path_member
                    / row["experiment"]
                    / f"version_{row['version']}"
                    / "test_task1.csv"
                )
                if pred_path_loc.is_file():
                    break
            if not pred_path_loc.is_file():
                true_j += 1
                continue

            pred = pd.read_csv(pred_path_loc)
            pred["trace_type_bin"] = pred["trace_type"] == "earthquake"

            fpr, tpr, thr = roc_curve(pred["trace_type_bin"], pred["score_detection"])
            auc = roc_auc_score(pred["trace_type_bin"], pred["score_detection"])
            idx = np.argmin(
                thr > row["det_threshold"]
            )  # Index of optimal threshold in terms of F1 score

            for ax in tmp_ax:
                ax.plot(fpr, tpr, label=model, color=f"C{true_j}", lw=3)
                ax.plot(fpr[idx], tpr[idx], "D", label=model, color=f"C{true_j}", ms=10)

            tmp_ax[-1].text(
                0.98,
                0.02 + true_j * lineheight,
                f"{MODEL_ABBREVIATIONS[model]} {auc:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=f"C{true_j}",
            )

            true_j += 1

        for ax in tmp_ax:
            ax.set_aspect("equal")
        tmp_ax[0].set_title(DATA_ALIASES[data])

        lim = ROC_LIMITS[data]
        tmp_ax[0].set_xlim(-lim / 50, lim)
        tmp_ax[0].set_ylim(1 - lim, 1 + lim / 50)

        if full_axis:
            tmp_ax[0].set_xlim(0, 1)
            tmp_ax[0].set_ylim(0, 1)

        if double_axis:
            tmp_ax[1].set_xlim(0, 1)
            tmp_ax[1].set_ylim(0, 1)

        true_i += 1

    for ax in axs[:, 0]:
        ax.set_ylabel("true positive rate")
    for ax in axs[-1]:
        ax.set_xlabel("false positive rate")

    return fig


def stead_snr(results, selection):
    stead = sbd.STEAD()

    results = results[results["data"] == "stead"]
    results = results[
        ~results["model"].isin(["cred", "gpd", "dppdetect", "dpppickers"])
    ]

    pred_path = [Path("pred"), Path("pred_baer")]

    fig = plt.figure(figsize=(5 * 3, 5 * 2))
    axs = fig.subplots(2, 3, sharex=True, sharey=True)

    for i, model in enumerate(sorted(results["model"].unique())):
        print(i, model)
        ax = axs[i // 3, i % 3]

        rows = results[results["model"] == model]
        row = rows.iloc[np.argmin(rows[selection])]

        for pred_path_member in pred_path:
            pred_path_loc = (
                pred_path_member
                / row["experiment"]
                / f"version_{row['version']}"
                / "test_task23.csv"
            )
            if pred_path_loc.is_file():
                break

            # For Bear picker, needs line without version
            pred_path_loc = pred_path_member / row["experiment"] / "test_task23.csv"
            if pred_path_loc.is_file():
                break

        pred = pd.read_csv(pred_path_loc)

        joined = pd.merge(pred, stead.metadata, on="trace_name", how="left")
        joined = joined[joined["phase_label"] == "P"]

        snr = joined["trace_snr_db"].values
        snr = np.array([float(x[1:-1].split()[0]) for x in snr])
        diff = (
            (joined["p_sample_pred"] - joined["phase_onset"]) / joined["sampling_rate"]
        ).values

        ax.hexbin(snr, diff, bins="log", extent=(-20, 100, -8, 8))
        ax.set_xlim(-15, 95)
        ax.set_ylim(-7, 7)
        ax.set_title(MODEL_ALIASES[model])

    for ax in axs[:, 0]:
        ax.set_ylabel("$t_{pred} - t_{true}~[s]$")
    for ax in axs[-1, :]:
        ax.set_xlabel("SNR [db]")

    return fig


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

    fig = plt.figure(figsize=(18, 18))

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
            n_bins = 61
            if true_i < sep0 and true_j < sep1 and adjust_scale:
                local_lim = 0.45
                n_bins = 46
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

                # For Bear picker, needs line without version
                pred_path_loc = pred_path_member / row["experiment"] / "test_task23.csv"
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

            if row["model"] == "baer":
                # Diff without early predictions as they are most likely outliers
                diff_reduced = diff[
                    pred_phase["p_sample_pred"] - pred_phase["start_sample"] > 100
                ]
            else:
                diff_reduced = diff

            axs[true_i, true_j].axvline(
                np.mean(diff_reduced), c="C1", lw=3, linestyle="--"
            )
            axs[true_i, true_j].axvline(
                np.median(diff_reduced), c="C3", lw=3, linestyle="--"
            )

            bins = np.linspace(-local_lim, local_lim, n_bins) + 1e-5
            axs[true_i, true_j].hist(diff, bins=bins)

            if ax1 == "model":
                title = MODEL_ALIASES[model]
            else:
                title = DATA_ALIASES[model]
            axs[0, true_j].set_title(title)

            outliers = np.abs(diff) > local_lim
            frac_outliers = np.sum(outliers) / len(diff)
            mae = np.mean(np.abs(diff_reduced))
            rmse = np.sqrt(np.mean(diff_reduced**2))

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
            if true_j == 0:
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
        ax.set_xlabel("$t_{pred} - t_{true}~[s]$")

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
    parser.add_argument(
        "--roc",
        action="store_true",
        help="If true, creates roc plot for detection.",
    )
    parser.add_argument(
        "--roc_cross",
        action="store_true",
        help="If true, creates roc cross plot for detection.",
    )
    parser.add_argument(
        "--phase_cross",
        action="store_true",
        help="If true, creates phase cross plots.",
    )
    parser.add_argument(
        "--thresholds",
        action="store_true",
        help="If true, creates thresholds tables.",
    )
    parser.add_argument(
        "--snr",
        action="store_true",
        help="If true, creates snr plot.",
    )

    args = parser.parse_args()
    main(
        args.base,
        args.cross,
        args.resampled,
        args.roc,
        args.roc_cross,
        args.phase_cross,
        args.thresholds,
        args.snr,
    )
