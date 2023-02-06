"""
This script offers general functionality required in multiple places.
"""

import numpy as np
import pandas as pd
import os
import logging
import packaging
import pytorch_lightning as pl


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

    if packaging.version.parse(pl.__version__) < packaging.version.parse("1.6.0"):
        checkpoint = f"epoch={min_row['epoch']:.0f}-step={min_row['step']:.0f}.ckpt"
    else:
        #  For default checkpoint filename, see https://github.com/Lightning-AI/lightning/pull/11805
        #  and https://github.com/Lightning-AI/lightning/issues/16636.
        #  epoch starts from 0, step starts from 0
        checkpoint = f"epoch={min_row['epoch']:.0f}-step={min_row['step']+1:.0f}.ckpt"

    if packaging.version.parse(pl.__version__) < packaging.version.parse("1.7.0"):
        version_id = version.split("_")[-1]
        version_str = f"{version_id}_{version_id}"

        checkpoint_path = (
            weights.parent
            / f"{weights.name}_{weights.name}"
            / version_str
            / "checkpoints"
            / checkpoint
        )
    else:
        # For default save path of checkpoints, see https://github.com/Lightning-AI/lightning/pull/12372
        # and https://lightning.ai/forums/t/how-to-get-the-checkpoint-path/327
        checkpoint_path = weights / version / "checkpoints" / checkpoint

    return model_cls.load_from_checkpoint(checkpoint_path)


default_workers = os.getenv("BENCHMARK_DEFAULT_WORKERS", None)
if default_workers is None:
    logging.warning(
        "BENCHMARK_DEFAULT_WORKERS not set. "
        "Will use 12 workers if not specified otherwise in configuration."
    )
    default_workers = 12
else:
    default_workers = int(default_workers)
