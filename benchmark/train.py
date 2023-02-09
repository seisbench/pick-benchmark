"""
This script handles the training of models base on model configuration files.
"""

import seisbench.generate as sbg
from seisbench.util import worker_seeding

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# https://github.com/Lightning-AI/lightning/pull/12554
# https://github.com/Lightning-AI/lightning/issues/11796
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
import packaging
import argparse
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import logging

import data
import models
from util import default_workers
import time
import datetime


def train(config, experiment_name, test_run):
    """
    Runs the model training defined by the config.

    Config parameters:

        - model: Model used as in the models.py file, but without the Lit suffix
        - data: Dataset used, as in seisbench.data
        - model_args: Arguments passed to the constructor of the model lightning module
        - trainer_args: Arguments passed to the lightning trainer
        - batch_size: Batch size for training and validation
        - num_workers: Number of workers for data loading.
          If not set, uses environment variable BENCHMARK_DEFAULT_WORKERS
        - restrict_to_phase: Filters datasets only to examples containing the given phase.
          By default, uses all phases.
        - training_fraction: Fraction of training blocks to use as float between 0 and 1. Defaults to 1.

    :param config: Configuration parameters for training
    :param test_run: If true, makes a test run with less data and less logging. Intended for debug purposes.
    """
    model = models.__getattribute__(config["model"] + "Lit")(
        **config.get("model_args", {})
    )

    train_loader, dev_loader = prepare_data(config, model, test_run)

    # CSV logger - also used for saving configuration as yaml
    csv_logger = CSVLogger("weights", experiment_name)
    csv_logger.log_hyperparams(config)
    loggers = [csv_logger]

    default_root_dir = os.path.join(
        "weights"
    )  # Experiment name is parsed from the loggers
    if not test_run:
        tb_logger = TensorBoardLogger("tb_logs", experiment_name)
        tb_logger.log_hyperparams(config)
        loggers += [tb_logger]

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, filename="{epoch}-{step}", monitor="val_loss", mode="min"
    )  # save_top_k=1, monitor="val_loss", mode="min": save the best model in terms of validation loss
    callbacks = [checkpoint_callback]

    ## Uncomment the following 2 lines to enable
    # device_stats = DeviceStatsMonitor()
    # callbacks.append(device_stats)

    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        logger=loggers,
        callbacks=callbacks,
        **config.get("trainer_args", {}),
    )

    trainer.fit(model, train_loader, dev_loader)


def prepare_data(config, model, test_run):
    """
    Returns the training and validation data loaders
    :param config:
    :param model:
    :param test_run:
    :return:
    """
    batch_size = config.get("batch_size", 1024)
    num_workers = config.get("num_workers", default_workers)
    dataset = data.get_dataset_by_name(config["data"])(
        sampling_rate=100, component_order="ZNE", dimension_order="NCW", cache="full"
    )
    restrict_to_phase = config.get("restrict_to_phase", None)
    if restrict_to_phase is not None:
        mask = generate_phase_mask(dataset, restrict_to_phase)
        dataset.filter(mask, inplace=True)

    if "split" not in dataset.metadata.columns:
        logging.warning("No split defined, adding auxiliary split.")
        split = np.array(["train"] * len(dataset))
        split[int(0.6 * len(dataset)) : int(0.7 * len(dataset))] = "dev"
        split[int(0.7 * len(dataset)) :] = "test"

        dataset._metadata["split"] = split

    train_data = dataset.train()
    dev_data = dataset.dev()

    if test_run:
        # Only use a small part of the dataset
        train_mask = np.zeros(len(train_data), dtype=bool)
        train_mask[:500] = True
        train_data.filter(train_mask, inplace=True)

        dev_mask = np.zeros(len(dev_data), dtype=bool)
        dev_mask[:500] = True
        dev_data.filter(dev_mask, inplace=True)

    training_fraction = config.get("training_fraction", 1.0)
    apply_training_fraction(training_fraction, train_data)

    train_data.preload_waveforms(pbar=True)
    dev_data.preload_waveforms(pbar=True)

    train_generator = sbg.GenericGenerator(train_data)
    dev_generator = sbg.GenericGenerator(dev_data)

    train_generator.add_augmentations(model.get_train_augmentations())
    dev_generator.add_augmentations(model.get_val_augmentations())

    train_loader = DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
        drop_last=True,  # Avoid crashes from batch norm layers for batch size 1
    )
    dev_loader = DataLoader(
        dev_generator,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )

    return train_loader, dev_loader


def apply_training_fraction(training_fraction, train_data):
    """
    Reduces the size of train_data to train_fraction by inplace filtering.
    Filter blockwise for efficient memory savings.

    :param training_fraction: Training fraction between 0 and 1.
    :param train_data: Training dataset
    :return: None
    """

    if not 0.0 < training_fraction <= 1.0:
        raise ValueError("Training fraction needs to be between 0 and 1.")

    if training_fraction < 1:
        blocks = train_data["trace_name"].apply(lambda x: x.split("$")[0])
        unique_blocks = blocks.unique()
        np.random.shuffle(unique_blocks)
        target_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
        target_blocks = set(target_blocks)
        mask = blocks.isin(target_blocks)
        train_data.filter(mask, inplace=True)


def generate_phase_mask(dataset, phases):
    mask = np.zeros(len(dataset), dtype=bool)

    for key, phase in models.phase_dict.items():
        if phase not in phases:
            continue
        else:
            if key in dataset.metadata:
                mask = np.logical_or(mask, ~np.isnan(dataset.metadata[key]))

    return mask


if __name__ == "__main__":
    code_start_time = time.perf_counter()

    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--lr", default=None, type=float)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    experiment_name = os.path.basename(args.config)[:-5]
    if args.lr is not None:
        logging.warning(f"Overwriting learning rate to {args.lr}")
        experiment_name += f"_{args.lr}"
        config["model_args"]["lr"] = args.lr

    if args.test_run:
        experiment_name = experiment_name + "_test"
    train(config, experiment_name, test_run=args.test_run)

    running_time = str(
        datetime.timedelta(seconds=time.perf_counter() - code_start_time)
    )
    print(f"Running time: {running_time}")
