"""
This script collects results in a folder, calculates performance metrics and writes them to csv.
"""

import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from tqdm import tqdm


def traverse_path(path, output, cross=False, resampled=False, baer=False):
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
        itr = exp_dir.iterdir()
        if baer:
            itr = [exp_dir]  # Missing version directory in the structure
        for version_dir in itr:
            if not version_dir.is_dir():
                pass

            results.append(
                process_version(
                    version_dir, cross=cross, resampled=resampled, baer=baer
                )
            )

    results = pd.DataFrame(results)
    if cross:
        sort_keys = ["data", "model", "target", "lr", "version"]
    else:
        sort_keys = ["data", "model", "lr", "version"]
    results.sort_values(sort_keys, inplace=True)
    results.to_csv(output, index=False)


def process_version(version_dir: Path, cross: bool, resampled: bool, baer: bool):
    """
    Extracts statistics for the given version of the given experiment.

    :param version_dir: Path to the specific version
    :param cross: If true, expects cross-domain results.
    :return: Results dictionary
    """
    stats = parse_exp_name(version_dir, cross=cross, resampled=resampled, baer=baer)

    stats.update(eval_task1(version_dir))
    stats.update(eval_task23(version_dir))

    return stats


def parse_exp_name(version_dir, cross, resampled, baer):
    if baer:
        exp_name = version_dir.name
        version = "0"
    else:
        exp_name = version_dir.parent.name
        version = version_dir.name.split("_")[-1]

    parts = exp_name.split("_")
    target = None
    sampling_rate = None
    if cross or baer:
        if len(parts) == 4:
            data, model, lr, target = parts
        else:
            data, model, target = parts
            lr = "0.001"
    elif resampled:
        if len(parts) == 5:
            data, model, lr, target, sampling_rate = parts
        else:
            data, model, target, sampling_rate = parts
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

    if cross or baer:
        stats["target"] = target
    if resampled:
        stats["target"] = target
        stats["sampling_rate"] = sampling_rate

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
    auc = roc_auc_score(dev_pred["trace_type_bin"], dev_pred["score_detection"])

    opt_index = np.nanargmax(f1)  # F1 optimal threshold index
    opt_thr = thr[opt_index]  # F1 optimal threshold value

    dev_stats = {
        "dev_det_precision": prec[opt_index],
        "dev_det_recall": recall[opt_index],
        "dev_det_f1": f1[opt_index],
        "dev_det_auc": auc,
        "det_threshold": opt_thr,
    }
    stats.update(dev_stats)

    prec, recall, f1, _ = precision_recall_fscore_support(
        test_pred["trace_type_bin"],
        test_pred["score_detection"] > opt_thr,
        average="binary",
    )
    auc = roc_auc_score(test_pred["trace_type_bin"], test_pred["score_detection"])
    test_stats = {
        "test_det_precision": prec,
        "test_det_recall": recall,
        "test_det_f1": f1,
        "test_det_auc": auc,
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

    def add_aux_columns(pred):
        for col in ["s_sample_pred", "score_p_or_s"]:
            if col not in pred.columns:
                pred[col] = np.nan

    add_aux_columns(dev_pred)
    add_aux_columns(test_pred)

    def nanmask(pred):
        """
        Returns all entries that are nan in score_p_or_s, p_sample_pred and s_sample_pred
        """
        mask = np.logical_and(
            np.isnan(pred["p_sample_pred"]), np.isnan(pred["s_sample_pred"])
        )
        mask = np.logical_and(mask, np.isnan(pred["score_p_or_s"]))
        return mask

    if nanmask(dev_pred).all():
        logging.warning(f"{version_dir} contains NaN predictions for tasks 2 and 3")
        return {}

    dev_pred = dev_pred[~nanmask(dev_pred)]
    test_pred = test_pred[~nanmask(test_pred)]

    skip_task2 = False
    if (
        np.logical_or(
            np.isnan(dev_pred["score_p_or_s"]), np.isinf(dev_pred["score_p_or_s"])
        ).all()
        or np.logical_or(
            np.isnan(test_pred["score_p_or_s"]), np.isinf(test_pred["score_p_or_s"])
        ).all()
    ):
        # For unfortunate combinations of nans and infs, otherwise weird scores can occur
        skip_task2 = True

    # Clipping removes infinitely likely P waves, usually resulting from models trained without S arrivals
    dev_pred["score_p_or_s"] = np.clip(dev_pred["score_p_or_s"].values, -1e100, 1e100)
    test_pred["score_p_or_s"] = np.clip(test_pred["score_p_or_s"].values, -1e100, 1e100)

    dev_pred_restricted = dev_pred[~np.isnan(dev_pred["score_p_or_s"])]
    test_pred_restricted = test_pred[~np.isnan(test_pred["score_p_or_s"])]
    if len(dev_pred_restricted) > 0 and not skip_task2:
        prec, recall, thr = precision_recall_curve(
            dev_pred_restricted["phase_label_bin"], dev_pred_restricted["score_p_or_s"]
        )

        f1 = 2 * prec * recall / (prec + recall)

        opt_index = np.nanargmax(f1)  # F1 optimal threshold index
        opt_thr = thr[opt_index]  # F1 optimal threshold value

        # Determine (approximately) optimal MCC threshold using 50 candidates
        mcc_thrs = np.sort(dev_pred["score_p_or_s"].values)
        mcc_thrs = mcc_thrs[np.linspace(0, len(mcc_thrs) - 1, 50, dtype=int)]
        mccs = []
        for thr in mcc_thrs:
            mccs.append(
                matthews_corrcoef(
                    dev_pred["phase_label_bin"], dev_pred["score_p_or_s"] > thr
                )
            )
        mcc = np.max(mccs)
        mcc_thr = mcc_thrs[np.argmax(mccs)]

        dev_stats = {
            "dev_phase_precision": prec[opt_index],
            "dev_phase_recall": recall[opt_index],
            "dev_phase_f1": f1[opt_index],
            "phase_threshold": opt_thr,
            "dev_phase_mcc": mcc,
            "phase_threshold_mcc": mcc_thr,
        }
        stats.update(dev_stats)

        prec, recall, f1, _ = precision_recall_fscore_support(
            test_pred_restricted["phase_label_bin"],
            test_pred_restricted["score_p_or_s"] > opt_thr,
            average="binary",
        )
        mcc = matthews_corrcoef(
            test_pred["phase_label_bin"], test_pred["score_p_or_s"] > mcc_thr
        )
        test_stats = {
            "test_phase_precision": prec,
            "test_phase_recall": recall,
            "test_phase_f1": f1,
            "test_phase_mcc": mcc,
        }
        stats.update(test_stats)

    for pred, set_str in [(dev_pred, "dev"), (test_pred, "test")]:
        for i, phase in enumerate(["P", "S"]):
            pred_phase = pred[pred["phase_label"] == phase]
            pred_col = f"{phase.lower()}_sample_pred"

            if len(pred_phase) == 0:
                continue

            diff = (pred_phase[pred_col] - pred_phase["phase_onset"]) / pred_phase[
                "sampling_rate"
            ]

            stats[f"{set_str}_{phase}_mean_s"] = np.mean(diff)
            stats[f"{set_str}_{phase}_std_s"] = np.sqrt(np.mean(diff**2))
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
    parser.add_argument(
        "--resampled",
        action="store_true",
        help="If true, expects cross-domain cross-sampling rate results.",
    )
    parser.add_argument(
        "--baer",
        action="store_true",
        help="If true, expects results from Baer-Kradolfer picker.",
    )

    args = parser.parse_args()

    traverse_path(
        args.path,
        args.output,
        cross=args.cross,
        resampled=args.resampled,
        baer=args.baer,
    )
