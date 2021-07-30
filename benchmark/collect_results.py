import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from tqdm import tqdm


def traverse_path(path, output, cross=False):
    """
    Traverses the given path and extracts results for each experiment and version

    :param path: Root path
    :param output: Path to write results csv to
    :param cross: If true, expects cross-domain results.
    :return: None
    """
    path = Path(path)

    results = []

    exp_dirs = [x for x in path.iterdir() if x.is_dir()]
    for exp_dir in tqdm(exp_dirs):
        for version_dir in exp_dir.iterdir():
            if not version_dir.is_dir():
                pass

            results.append(process_version(version_dir, cross=cross))

    results = pd.DataFrame(results)
    if cross:
        sort_keys = ["data", "model", "target", "lr", "version"]
    else:
        sort_keys = ["data", "model", "lr", "version"]
    results.sort_values(sort_keys, inplace=True)
    results.to_csv(output, index=False)


def process_version(version_dir: Path, cross: bool):
    """
    Extracts statistics for the given version of the given experiment.

    :param version_dir: Path to the specific version
    :param cross: If true, expects cross-domain results.
    :return: Results dictionary
    """
    stats = parse_exp_name(version_dir, cross=cross)

    stats.update(eval_task1(version_dir))
    stats.update(eval_task23(version_dir))

    return stats


def parse_exp_name(version_dir, cross):
    exp_name = version_dir.parent.name
    version = version_dir.name.split("_")[-1]

    parts = exp_name.split("_")
    target = None
    if cross:
        if len(parts) == 4:
            data, model, lr, target = parts
        else:
            data, model, target = parts
            lr = "0.001"
    else:
        if len(parts) == 3:
            data, model, lr = parts
        else:
            data, model = parts
            lr = "0.001"

    lr = float(lr)

    stats = {
        "experiment": exp_name,
        "data": data,
        "model": model,
        "lr": lr,
        "version": version,
    }

    if cross:
        stats["target"] = target

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
    # Clipping removes infinitely likely P waves, usually resulting from models trained without S arrivals
    dev_pred["score_p_or_s"] = np.clip(dev_pred["score_p_or_s"].values, -1e100, 1e100)
    test_pred["score_p_or_s"] = np.clip(test_pred["score_p_or_s"].values, -1e100, 1e100)

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
        "phase_threshold": opt_thr,
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
            stats[f"{set_str}_{phase}_std_s"] = np.sqrt(np.mean(diff ** 2))
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
    parser.add_argument(
        "--cross", action="store_true", help="If true, expects cross-domain results."
    )

    args = parser.parse_args()

    traverse_path(args.path, args.output, cross=args.cross)
