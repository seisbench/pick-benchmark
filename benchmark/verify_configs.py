"""
This script performs a list of checks on a folder of config files.
Please see the command line argument description for details on the checks.
"""

import argparse
from pathlib import Path
import logging
import json


def check_configs(path):
    path = Path(path)

    datasets = {}
    models = {}
    for config_path in path.iterdir():
        if not config_path.name.endswith(".json"):
            continue

        try:
            dataset, model = config_path.name.split("_")
        except:
            logging.warning(f"Unexpected config naming scheme {config_path.name}.")
            dataset, model = None, None

        with open(config_path, "r") as f:
            config = json.load(f)

        if dataset in datasets:
            if datasets[dataset] != config["data"]:
                logging.warning(
                    f"{config_path.name}: Expected {datasets[dataset]}, found {config['data']}"
                )
        else:
            datasets[dataset] = config["data"]

        if model in models:
            if models[model] != config["model"]:
                logging.warning(
                    f"{config_path.name}: Expected {models[model]}, found {config['model']}"
                )
        else:
            models[model] = config["model"]

        if "sigma" in config["model_args"]:
            logging.warning(f"Found sigma in {config_path.name}")

        if config["model_args"]["lr"] != 1e-3:
            logging.warning(f"Found unexpected learning rate in {config_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checks a folder with configurations. "
        "Checks that: "
        "(1) No config contains a sigma value. "
        "(2) All learning rates are the same. "
        "(3) Naming is consistent with datasets and models."
    )
    parser.add_argument("path", type=str, help="Path to the configuration folder.")
    args = parser.parse_args()

    check_configs(args.path)
