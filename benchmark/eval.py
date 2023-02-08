"""
This script implements functionality for evaluating models.
Given a model and a set of targets, it calculates and outputs predictions.
"""

import seisbench.generate as sbg

import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

import models
import data
import logging
from util import load_best_model, default_workers
import time
import datetime
import packaging

data_aliases = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "stead": "STEAD",
    "neic": "NEIC",
    "instance": "InstanceCountsCombined",
    "iquique": "Iquique",
    "lendb": "LenDB",
    "scedc": "SCEDC",
}


def main(weights, targets, sets, batchsize, num_workers, sampling_rate=None):
    weights = Path(weights)
    targets = Path(targets)
    sets = sets.split(",")

    version = sorted(weights.iterdir())[-1]
    config_path = version / "hparams.yaml"
    with open(config_path, "r") as f:
        # config = yaml.safe_load(f)
        config = yaml.full_load(f)

    model_cls = models.__getattribute__(config["model"] + "Lit")
    model = load_best_model(model_cls, weights, version.name)

    data_name = data_aliases[targets.name]

    if data_name != config["data"]:
        logging.warning("Detected cross-domain evaluation")
        pred_root = "pred_cross"
        parts = weights.name.split()
        weight_path_name = "_".join(parts[:2] + [targets.name] + parts[2:])

    else:
        pred_root = "pred"
        weight_path_name = weights.name

    dataset = data.get_dataset_by_name(data_name)(
        sampling_rate=100, component_order="ZNE", dimension_order="NCW", cache="full"
    )

    if sampling_rate is not None:
        dataset.sampling_rate = sampling_rate
        pred_root = pred_root + "_resampled"
        weight_path_name = weight_path_name + f"_{sampling_rate}"

    for eval_set in sets:
        split = dataset.get_split(eval_set)
        if targets.name == "instance":
            logging.warning(
                "Overwriting noise trace_names to allow correct identification"
            )
            # Replace trace names for noise entries
            split._metadata["trace_name"].values[
                -len(split.datasets[-1]) :
            ] = split._metadata["trace_name"][-len(split.datasets[-1]) :].apply(
                lambda x: "noise_" + x
            )
            split._build_trace_name_to_idx_dict()

        logging.warning(f"Starting set {eval_set}")
        split.preload_waveforms(pbar=True)

        for task in ["1", "23"]:
            task_csv = targets / f"task{task}.csv"

            if not task_csv.is_file():
                continue

            logging.warning(f"Starting task {task}")

            task_targets = pd.read_csv(task_csv)
            task_targets = task_targets[task_targets["trace_split"] == eval_set]
            if task == "1" and targets.name == "instance":
                border = _identify_instance_dataset_border(task_targets)
                task_targets["trace_name"].values[border:] = task_targets["trace_name"][
                    border:
                ].apply(lambda x: "noise_" + x)

            if sampling_rate is not None:
                for key in ["start_sample", "end_sample", "phase_onset"]:
                    if key not in task_targets.columns:
                        continue
                    task_targets[key] = (
                        task_targets[key]
                        * sampling_rate
                        / task_targets["sampling_rate"]
                    )
                task_targets[sampling_rate] = sampling_rate

            restrict_to_phase = config.get("restrict_to_phase", None)
            if restrict_to_phase is not None and "phase_label" in task_targets.columns:
                mask = task_targets["phase_label"].isin(list(restrict_to_phase))
                task_targets = task_targets[mask]

            if restrict_to_phase is not None and task == "1":
                logging.warning("Skipping task 1 as restrict_to_phase is set.")
                continue

            generator = sbg.SteeredGenerator(split, task_targets)
            generator.add_augmentations(model.get_eval_augmentations())

            loader = DataLoader(
                generator, batch_size=batchsize, shuffle=False, num_workers=num_workers
            )
            trainer = pl.Trainer(accelerator="gpu", devices=1)

            predictions = trainer.predict(model, loader)

            # Merge batches
            merged_predictions = []
            for i, _ in enumerate(predictions[0]):
                merged_predictions.append(torch.cat([x[i] for x in predictions]))

            merged_predictions = [x.cpu().numpy() for x in merged_predictions]
            task_targets["score_detection"] = merged_predictions[0]
            task_targets["score_p_or_s"] = merged_predictions[1]
            task_targets["p_sample_pred"] = (
                merged_predictions[2] + task_targets["start_sample"]
            )
            task_targets["s_sample_pred"] = (
                merged_predictions[3] + task_targets["start_sample"]
            )

            pred_path = (
                weights.parent.parent
                / pred_root
                / weight_path_name
                / version.name
                / f"{eval_set}_task{task}.csv"
            )
            pred_path.parent.mkdir(exist_ok=True, parents=True)
            task_targets.to_csv(pred_path, index=False)


def _identify_instance_dataset_border(task_targets):
    """
    Calculates the dataset border between Signal and Noise for instance,
    assuming it is the only place where the bucket number does not increase
    """
    buckets = task_targets["trace_name"].apply(lambda x: int(x.split("$")[0][6:]))

    last_bucket = 0
    for i, bucket in enumerate(buckets):
        if bucket < last_bucket:
            return i
        last_bucket = bucket


if __name__ == "__main__":
    code_start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model using a set of targets."
    )
    parser.add_argument(
        "weights",
        type=str,
        help="Path to weights. "
        "The script will automatically load the configuration and the model. "
        "The script always uses the newest version and the checkpoint with lowest validation loss."
        "Predictions will be written into the weights path as csv."
        "Note: Due to pytorch lightning internals, there exist two weights folders, "
        "{weights} and {weight}_{weights}. Please use the former as parameter",
    )
    parser.add_argument(
        "targets",
        type=str,
        help="Path to evaluation targets folder. "
        "The script will detect which tasks are present base on file names.",
    )
    parser.add_argument(
        "--sets",
        type=str,
        default="dev,test",
        help="Sets on which to evaluate, separated by commata. Defaults to dev and test.",
    )
    parser.add_argument("--batchsize", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--num_workers",
        default=default_workers,
        help="Number of workers for data loader",
    )
    parser.add_argument(
        "--sampling_rate", type=float, help="Overwrites the sampling rate in the data"
    )
    args = parser.parse_args()

    main(
        args.weights,
        args.targets,
        args.sets,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        sampling_rate=args.sampling_rate,
    )
    running_time = str(
        datetime.timedelta(seconds=time.perf_counter() - code_start_time)
    )
    print(f"Running time: {running_time}")
