"""
This script is used to export models into the SeisBench format and folder structure.
This allows used through "from_pretrained".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import torch
import json
import copy

import models
from util import load_best_model
from results_summary import DATA_ALIASES

json_base = {
    "docstring": "Model trained on DATASET for 100 epochs with a learning rate of LR.\n"
    "Threshold selected for optimal F1 score on in-domain evaluation. "
    "Depending on the target region, the thresholds might need to be adjusted.\n"
    "When using this model, please reference the SeisBench publications listed "
    "at https://github.com/seisbench/seisbench\n\n"
    "Jannes Münchmeyer, Jack Woollam (munchmej@gfz-potsdam.de, jack.woollam@kit.edu)",
    "model_args": {
        "component_order": "ZNE",
        "norm": "peak",
    },
    "seisbench_requirement": "0.3.0",
    "version": "1",
}


def main():
    full_res = pd.read_csv("results.csv")

    full_res = full_res[full_res["model"] != "gpd"].copy()
    full_res["model"].replace("gpdpick", "gpd", inplace=True)

    for pair, subdf in tqdm(full_res.groupby(["data", "model"])):
        idx = get_optimal_model_idx(subdf)
        if idx is None:
            print(f"Skipping {pair}")
            continue

        export_model(subdf.iloc[idx])


def get_optimal_model_idx(subdf):
    """
    Identifies the optimal model among the candidates in subdf.
    The optimal model is determined as the model with the lowest average relative loss in the metrics.

    Example:
        Model 1: det_auc=1 phase_mcc=0.9
        Model 2: det_auc=0.98 phase_mcc = 1
        Here we will select model 2, because model 1 on average only achieves a performance of 0.95 compared to the
        optimum, but model 2 achieves 0.99.

    In contrast to the example, the model also takes P and S std into account.

    :param subdf:
    :return: idx or None if no model is valid
    """
    x = subdf[
        ["dev_det_auc", "dev_phase_mcc", "dev_P_std_s", "dev_S_std_s"]
    ].values.copy()
    x[:, 2:] = 1 / x[:, 2:]
    x /= np.max(x, axis=0, keepdims=True)
    means = np.nanmean(x, axis=1)
    if np.isnan(means).all():
        return None

    return np.nanargmax(means)


def generate_metadata(row):
    meta = copy.deepcopy(json_base)
    default_args = {}
    meta["docstring"] = meta["docstring"].replace("DATASET", DATA_ALIASES[row["data"]])
    meta["docstring"] = meta["docstring"].replace("LR", str(row["lr"]))
    if row["model"] in ["cred", "eqtransformer"]:
        det_threshold = row["det_threshold"]
        if np.isnan(det_threshold):
            det_threshold = (
                0.3  # Roughly the average detection threshold across datasets
            )
        default_args["detection_threshold"] = det_threshold
        if row["model"] == "eqtransformer":
            # As the outputs are independent, and the empirical phase_thresholds are usually close to 1,
            # we just suggest the detection threshold for each phase as well.
            default_args["P_threshold"] = det_threshold
            default_args["S_threshold"] = det_threshold

    elif row["model"] in ["dpppickerp", "dpppickers"]:
        pass

    elif row["model"] in ["phasenet", "basicphaseae", "dppdetect"]:
        meta["model_args"]["phases"] = "PSN"
        det_threshold = row["det_threshold"]
        if np.isnan(det_threshold):
            det_threshold = 0.4
        phase_threshold = row["phase_threshold"]
        if np.isnan(phase_threshold):
            phase_threshold = 1
        default_args["P_threshold"] = det_threshold * np.sqrt(phase_threshold)
        default_args["S_threshold"] = det_threshold / np.sqrt(phase_threshold)

    elif row["model"] == "gpd":
        meta["model_args"]["phases"] = "PSN"
        meta["model_args"]["filter_args"] = ["highpass"]
        meta["model_args"]["filter_kwargs"] = {"freq": 0.5}
        det_threshold = row["det_threshold"]
        if np.isnan(det_threshold):
            det_threshold = 0.8
        phase_threshold = row["phase_threshold"]
        if np.isnan(phase_threshold):
            phase_threshold = 1
        default_args["P_threshold"] = det_threshold * np.sqrt(phase_threshold)
        default_args["S_threshold"] = det_threshold / np.sqrt(phase_threshold)

    else:
        raise ValueError("Unknown model type")

    if row["model"] == "phasenet":
        default_args["blinding"] = [250, 250]

    meta["default_args"] = default_args

    return meta


def export_model(row):
    output_base = Path("seisbench_models")
    weights = Path("weights") / row["experiment"]

    version = sorted(weights.iterdir())[-1]
    config_path = version / "hparams.yaml"
    with open(config_path, "r") as f:
        # config = yaml.safe_load(f)
        config = yaml.full_load(f)

    model_cls = models.__getattribute__(config["model"] + "Lit")
    model = load_best_model(model_cls, weights, version.name)

    output_path = output_base / row["model"] / f"{row['data']}.pt.v1"
    json_path = output_base / row["model"] / f"{row['data']}.json.v1"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), output_path)

    meta = generate_metadata(row)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
