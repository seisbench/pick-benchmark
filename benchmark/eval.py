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


data_aliases = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "stead": "STEAD",
    "neic": "NEIC",
    "instance": "InstanceCountsCombined",
    "lendb": "LenDB",
    "scedc": "SCEDC",
}


def main(weights, targets, sets, batchsize, num_workers):
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

    for eval_set in sets:
        split = dataset.get_split(eval_set)
        logging.warning(f"Starting set {eval_set}")
        split.preload_waveforms(pbar=True)

        for task in ["1", "23"]:
            task_csv = targets / f"task{task}.csv"

            if not task_csv.is_file():
                continue

            logging.warning(f"Starting task {task}")

            task_targets = pd.read_csv(task_csv)
            task_targets = task_targets[task_targets["trace_split"] == eval_set]

            generator = sbg.SteeredGenerator(split, task_targets)
            generator.add_augmentations(model.get_eval_augmentations())

            loader = DataLoader(
                generator, batch_size=batchsize, shuffle=False, num_workers=num_workers
            )

            trainer = pl.Trainer(gpus=1)

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


def load_best_model(model_cls, weights, version):
    """
    Determines the model with lowest validation loss from the csv logs and loads it

    :param model_cls: Class of the lightning module to load
    :param weights: Path to weights as in cmd arguments
    :param version: String of version file
    :return: Instance of lightning module that was loaded from the best checkpoint
    """
    metrics = pd.read_csv(weights / version / "metrics.csv")

    idx = np.nanargmin(metrics["val_loss"])
    min_row = metrics.iloc[idx]

    checkpoint = f"epoch={min_row['epoch']:.0f}-step={min_row['step']:.0f}.ckpt"

    version_id = version.split("_")[-1]
    version_str = f"{version_id}_{version_id}"

    checkpoint_path = (
        weights.parent
        / f"{weights.name}_{weights.name}"
        / version_str
        / "checkpoints"
        / checkpoint
    )

    return model_cls.load_from_checkpoint(checkpoint_path)


if __name__ == "__main__":
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
        "--num_workers", default=12, help="Number of workers for data loader"
    )
    args = parser.parse_args()

    main(
        args.weights,
        args.targets,
        args.sets,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
    )
