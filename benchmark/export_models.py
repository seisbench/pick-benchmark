import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import torch

import models
from util import load_best_model


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

    output_path = output_base / row["model"] / f"{row['data']}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), output_path)
    # TODO: Write json files


if __name__ == "__main__":
    main()
