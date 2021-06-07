import seisbench.generate as sbg

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
import json
import numpy as np
from torch.utils.data import DataLoader
import os

import data
import models


def train(config, experiment_name, test_run):
    """
    Runs the model training defined by the config.

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

    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        logger=loggers,
        **config.get("trainer_args", {})
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
    num_workers = config.get("num_workers", 12)
    dataset = data.get_dataset_by_name(config["data"])(
        sampling_rate=100, component_order="ZNE", dimension_order="NCW", cache="full"
    )

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

    train_data.preload_waveforms(pbar=True)
    dev_data.preload_waveforms(pbar=True)

    train_generator = sbg.GenericGenerator(train_data)
    dev_generator = sbg.GenericGenerator(dev_data)

    train_generator.add_augmentations(model.get_augmentations())
    dev_generator.add_augmentations(model.get_augmentations())

    train_loader = DataLoader(
        train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    dev_loader = DataLoader(
        dev_generator, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, dev_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    experiment_name = os.path.basename(args.config)[:-5]
    train(config, experiment_name, test_run=args.test_run)
