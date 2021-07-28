import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support


def traverse_path(path, output):
    """
    Traverses the given path and extracts results for each experiment and version

    :param path: Root path
    :param output: Path to write results csv to
    :return: None
    """
    path = Path(path)

    results = []

    for exp_dir in path.iterdir():
        if not exp_dir.is_dir():
            pass

        for version_dir in exp_dir.iterdir():
            if not version_dir.is_dir():
                pass

            results.append(process_version(version_dir))

    results = pd.DataFrame(results)
    results.sort_values(["data", "model"], inplace=True)
    results.to_csv(output, index=False)


def process_version(version_dir: Path):
    """
    Extracts statistics for the given version of the given experiment.

    :param version_dir: Path to the specific version
    :return: Results dictionary
    """

    exp_name = version_dir.parent.name
    data, model = exp_name.split("_")
    version = version_dir.name.split("_")[-1]
    stats = {"experiment": exp_name, "data": data, "model": model, "version": version}

    stats.update(eval_task1(version_dir))
    stats.update(eval_task23(version_dir))

    return stats


def eval_task1(version_dir: Path):
    if not (
        (version_dir / "dev_task1.csv").is_file()
        and (version_dir / "test_task1.csv").is_file()
    ):
        logging.warning(f"Directory {version_dir} does not contain task 1")
        return {}

    stats = {}

    dev_pred = pd.read_csv(version_dir / "dev_task1.csv")
    dev_pred["trace_type_bin"] = dev_pred["trace_type"] == "earthquake"
    test_pred = pd.read_csv(version_dir / "test_task1.csv")
    test_pred["trace_type_bin"] = test_pred["trace_type"] == "earthquake"

    prec, recall, thr = precision_recall_curve(
        dev_pred["trace_type_bin"], dev_pred["score_detection"]
    )

    f1 = 2 * prec * recall / (prec + recall)

    opt_index = np.nanargmax(f1)  # F1 optimal threshold index
    opt_thr = thr[opt_index]  # F1 optimal threshold value

    dev_stats = {
        "dev_det_precision": prec[opt_index],
        "dev_det_recall": recall[opt_index],
        "dev_det_f1": f1[opt_index],
        "det_threshold": opt_thr,
    }
    stats.update(dev_stats)

    prec, recall, f1, _ = precision_recall_fscore_support(
        test_pred["trace_type_bin"],
        test_pred["score_detection"] > opt_thr,
        average="binary",
    )
    test_stats = {
        "test_det_precision": prec,
        "test_det_recall": recall,
        "test_det_f1": f1,
    }
    stats.update(test_stats)

    return stats


def eval_task23(version_dir: Path):
    if not (
        (version_dir / "dev_task23.csv").is_file()
        and (version_dir / "test_task23.csv").is_file()
    ):
        logging.warning(f"Directory {version_dir} does not contain tasks 2 and 3")
        return {}

    stats = {}

    dev_pred = pd.read_csv(version_dir / "dev_task23.csv")
    dev_pred["phase_label_bin"] = dev_pred["phase_label"] == "P"
    test_pred = pd.read_csv(version_dir / "test_task23.csv")
    test_pred["phase_label_bin"] = test_pred["phase_label"] == "P"

    if np.isnan(dev_pred["score_p_or_s"]).all():
        logging.warning(f"{version_dir} contains NaN predictions for tasks 2 and 3")
        return {}

    dev_pred = dev_pred[~np.isnan(dev_pred["score_p_or_s"])]
    test_pred = test_pred[~np.isnan(test_pred["score_p_or_s"])]

    prec, recall, thr = precision_recall_curve(
        dev_pred["phase_label_bin"], dev_pred["score_p_or_s"]
    )

    f1 = 2 * prec * recall / (prec + recall)

    opt_index = np.nanargmax(f1)  # F1 optimal threshold index
    opt_thr = thr[opt_index]  # F1 optimal threshold value

    dev_stats = {
        "dev_phase_precision": prec[opt_index],
        "dev_phase_recall": recall[opt_index],
        "dev_phase_f1": f1[opt_index],
        "det_threshold": opt_thr,
    }
    stats.update(dev_stats)

    prec, recall, f1, _ = precision_recall_fscore_support(
        test_pred["phase_label_bin"],
        test_pred["score_p_or_s"] > opt_thr,
        average="binary",
    )
    test_stats = {
        "test_phase_precision": prec,
        "test_phase_recall": recall,
        "test_phase_f1": f1,
    }
    stats.update(test_stats)

    for pred, set_str in [(dev_pred, "dev"), (test_pred, "test")]:
        for i, phase in enumerate(["P", "S"]):
            pred_phase = pred[pred["phase_label"] == phase]
            pred_col = f"{phase.lower()}_sample_pred"

            diff = (pred_phase[pred_col] - pred_phase["phase_onset"]) / pred_phase[
                "sampling_rate"
            ]

            stats[f"{set_str}_{phase}_mean_s"] = np.mean(diff)
            stats[f"{set_str}_{phase}_std_s"] = np.std(diff)
            stats[f"{set_str}_{phase}_mae_s"] = np.mean(np.abs(diff))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collects results from all experiments in a folder and outputs them in condensed csv format."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Root path of predictions",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path for the output csv",
    )

    args = parser.parse_args()

    traverse_path(args.path, args.output)
