"""
This file implements training and evaluation of a Baer-Kradolfer picker for P picks on SeisBench datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seisbench.generate as sbg
from tqdm import tqdm
from obspy.signal.trigger import pk_baer
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
import argparse
import os
import json
import logging

import data


data_aliases = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "stead": "STEAD",
    "neic": "NEIC",
    "instance": "InstanceCounts",
    "iquique": "Iquique",
    "lendb": "LenDB",
    "scedc": "SCEDC",
}


class BaerKradolfer:
    def __init__(self, lp, hp, tdownmax, tupevent, thr1, windowlens=(6000, 2000, 1000)):
        self.lp = lp
        self.hp = hp
        self.tdownmax = tdownmax
        self.tupevent = tupevent
        self.thr1 = thr1
        self.windowlens = windowlens

    def get_augmentations(self):
        hp = self.hp
        lp = max(self.lp, self.hp + 0.01)  # Make sure lowpass is always above highpass

        windows = [
            sbg.SteeredWindow(
                windowlen=windowlen,
                strategy="variable",
                key=("X", f"X{i}"),
                window_output_key=f"window_borders{i}",
            )
            for i, windowlen in enumerate(self.windowlens)
        ]

        return [
            sbg.Normalize(detrend_axis=-1),
            sbg.Filter(2, (hp, lp), "bandpass"),
        ] + windows

    def predict(self, sample):
        pred_relative_to_p0 = -1
        window_idx = 0

        # Shrink window until pick falls into window
        while (
            window_idx < len(self.windowlens) and not 0 <= pred_relative_to_p0 <= 1000
        ):
            x = sample[f"X{window_idx}"][0]  # Remove channel axis
            p0, p1 = sample[f"window_borders{window_idx}"]
            p_pick, _ = pk_baer(
                x,
                100,
                int(100 * self.tdownmax),
                int(100 * self.tupevent),
                self.thr1,
                2 * self.thr1,
                100,
                100,
            )
            pred_relative_to_p0 = p_pick - p0
            window_idx += 1

        return pred_relative_to_p0

    @classmethod
    def load_from_log(cls, path):
        opt = -np.inf
        params = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    pass
                parsed_line = json.loads(line)
                if parsed_line["target"] > opt:
                    opt = parsed_line["target"]
                    params = parsed_line["params"]

        logging.warning(f"Optimal value: {- opt:.1f}\tParams: {params}")
        return cls(**params)


def train(target, pbar, limit):
    np.random.seed(42)
    targets = Path("targets") / target / "task23.csv"
    targets = pd.read_csv(targets)
    targets = targets[targets["trace_split"] == "dev"]
    targets = targets[targets["phase_label"] == "P"]

    trace_names = None
    if len(targets) > limit:
        trace_names = targets["trace_name"].values.copy()
        np.random.shuffle(trace_names)
        trace_names = trace_names[:limit]
        mask = targets["trace_name"].isin(set(trace_names))
        targets = targets[mask]

    dataset = data.get_dataset_by_name(data_aliases[target])(
        sampling_rate=100, component_order="Z", dimension_order="NCW", cache="trace"
    )
    dataset = dataset.dev()
    if trace_names is not None:
        mask = dataset.metadata["trace_name"].isin(set(trace_names))
        dataset.filter(mask, inplace=True)
    dataset.preload_waveforms(pbar=True)

    bounds = {
        "lp": (1, 49),
        "hp": (0.001, 5),
        "tdownmax": (0.5, 15),
        "tupevent": (0.3, 3),
        "thr1": (1, 40),
    }

    picker = BaerKradolfer(0, 0, 0, 0, 0)  # Parameters will anyhow be overwritten

    def fitness(lp, hp, tdownmax, tupevent, thr1):
        """
        Fitness function for Bayesian optimization

        :param lp: Lowpass frequency
        :param hp: Highpass frequency
        :param tdownmax: See obspy.signal.trigger.pk_bear
        :param tupevent: See obspy.signal.trigger.pk_bear
        :param thr1: See obspy.signal.trigger.pk_bear . We also set thr2 = 2 * thr1
        :return:
        """
        picker.lp = lp
        picker.hp = hp
        picker.tdownmax = tdownmax
        picker.tupevent = tupevent
        picker.thr1 = thr1

        generator = sbg.SteeredGenerator(dataset, targets)
        generator.add_augmentations(picker.get_augmentations())

        preds = []
        if pbar:
            itr = tqdm(range(len(generator)), total=len(generator))
        else:
            itr = range(len(generator))

        for i in itr:
            pred_relative_to_p0 = picker.predict(generator[i])
            row = targets.iloc[i]

            pred = row["start_sample"] + pred_relative_to_p0
            preds.append(pred)

        preds = np.array(preds)
        rmse = np.sqrt(
            np.mean((preds - targets["phase_onset"].values) ** 2)
        )  # RMSE in samples

        return (
            -rmse
        )  # Negative as optimizer is built for maximizing the target function

    optimizer = BayesianOptimization(
        f=fitness,
        pbounds=bounds,
        random_state=1,
    )

    # Setup loggers
    os.makedirs("baer_logs", exist_ok=True)
    logger = JSONLogger(path=f"baer_logs/{target}.json")
    screen_logger = ScreenLogger()
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, screen_logger)

    # Roughly matching entries from NMSOP
    optimizer.probe(
        params={"lp": 49, "hp": 3, "tdownmax": 2, "tupevent": 1.5, "thr1": 10},
        lazy=True,
    )
    optimizer.probe(
        params={"lp": 5, "hp": 1, "tdownmax": 3, "tupevent": 1.5, "thr1": 5}, lazy=True
    )
    optimizer.probe(
        params={"lp": 2, "hp": 0.1, "tdownmax": 5, "tupevent": 1.5, "thr1": 2},
        lazy=True,
    )

    optimizer.maximize(init_points=25, n_iter=500)

    print(optimizer.max)


def eval(source, target):
    targets = Path("targets") / target / "task23.csv"
    targets = pd.read_csv(targets)
    targets = targets[targets["phase_label"] == "P"]

    dataset = data.get_dataset_by_name(data_aliases[target])(
        sampling_rate=100, component_order="Z", dimension_order="NCW", cache=None
    )  # Caching is disabled to save memory. We only expect a relatively minor impact on speed in this case.

    model_path = Path("baer_logs") / f"{source}.json"
    model = BaerKradolfer.load_from_log(model_path)

    for eval_set in ["dev", "test"]:
        logging.warning(f"Starting set {eval_set}")
        split = dataset.get_split(eval_set)
        split.preload_waveforms(pbar=True)

        split_targets = targets[targets["trace_split"] == eval_set].copy()

        generator = sbg.SteeredGenerator(split, split_targets)
        generator.add_augmentations(model.get_augmentations())

        preds = []
        itr = tqdm(range(len(generator)), total=len(generator))

        for i in itr:
            pred_relative_to_p0 = model.predict(generator[i])
            preds.append(pred_relative_to_p0)

        split_targets["p_sample_pred"] = preds + split_targets["start_sample"]

        pred_path = (
            Path("pred_baer") / f"{source}_baer_{target}" / f"{eval_set}_task23.csv"
        )
        pred_path.parent.mkdir(exist_ok=True, parents=True)
        split_targets.to_csv(pred_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--source", type=str)
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("--limit", type=int, default=2500)
    args = parser.parse_args()

    if args.action == "train":
        train(args.target, pbar=args.pbar, limit=args.limit)

    if args.action == "eval":
        eval(source=args.source, target=args.target)
