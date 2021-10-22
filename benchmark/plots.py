"""
This file offers generic plotting functions.
Please note that some plotting functions are also contained directly in results_summary.py.
"""

from pathlib import Path
import pandas as pd
import seisbench.generate as sbg
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal

import benchmark.models as models
from benchmark.util import load_best_model


color_dict = {"det": "C3", "P": "C4", "S": "C5"}
pred_lw = 3


def basicphaseae_plot(idx, ax, model, generator):
    batch = generator[idx]
    x = torch.tensor(batch["X"]).unsqueeze(0)
    p0, p1 = batch["window_borders"]

    re = torch.zeros(x.shape[:2] + (7, 600), dtype=x.dtype)
    for i, start in enumerate(range(0, 2401, 400)):
        re[:, :, i] = x[:, :, start : start + 600]

    x = re
    x = x.permute(0, 2, 1, 3)  # --> (batch, windows, channels, samples)
    shape_save = x.shape
    x = x.reshape(-1, 3, 600)  # --> (batch * windows, channels, samples)
    pred = model(x.to(model.device))
    pred = pred.reshape(
        shape_save[:2] + (3, 600)
    )  # --> (batch, window, channels, samples)
    full_pred = torch.zeros((pred.shape[0], pred.shape[2], 3000))
    for i, start in enumerate(range(0, 2401, 400)):
        if start == 0:
            # Use full window (for start==0, the end will be overwritten)
            full_pred[:, :, start : start + 600] = pred[:, i]
        else:
            full_pred[:, :, start + 100 : start + 600] = pred[:, i, :, 100:]

    # ax.plot(1 - pred[0, 2, p0:p1])
    ax.plot(full_pred[0, 0, p0:p1], color_dict["P"], lw=pred_lw)
    ax.plot(full_pred[0, 1, p0:p1], color_dict["S"], lw=pred_lw)

    row = generator.metadata.iloc[idx]
    if "phase_label" not in row:
        return

    phase_id = {"P": 0, "S": 1}[row["phase_label"]]

    ax.axvline(
        np.argmax(full_pred[0, phase_id, p0:p1]),
        c=color_dict[row["phase_label"]],
        lw=pred_lw,
    )


def cred_plot(idx, ax, model, generator):
    batch = generator[idx]
    x = torch.tensor(batch["spec"]).unsqueeze(0)
    p0, p1 = batch["window_borders"]

    samples = p1 - p0

    pred = model(x.to(model.device)).cpu()
    p0 = p0 // 158
    p1 = p1 // 158 + 1
    pred = pred[0, p0:p1, 0]
    ax.plot(np.linspace(0, samples, pred.shape[0]), pred, color_dict["det"], lw=pred_lw)


def dpp_plot(idx, ax, models, generator):
    try:
        batch = generator[idx]
    except ValueError:
        return
    row = generator.metadata.iloc[idx]
    if "phase_label" not in row:
        return

    if row["phase_label"] == "P":
        dpp = models[0]
        c = color_dict["P"]
    else:
        dpp = models[1]
        c = color_dict["S"]

    x = torch.tensor(batch["X"]).unsqueeze(0)
    pred = dpp(x.to(dpp.device)).cpu()
    ax.plot(pred[0], c=c, lw=pred_lw)
    ax.axvline(np.argmax(pred[0] > 0.5), c=c, lw=pred_lw)


def eqtransformer_plot(idx, ax, model, generator):
    batch = generator[idx]
    p0, p1 = batch["window_borders"]
    det_pred, p_pred, s_pred = model(
        torch.tensor(batch["X"], device=model.device).unsqueeze(0)
    )
    ax.plot(det_pred[0, p0:p1].cpu(), color_dict["det"], lw=pred_lw)
    ax.plot(p_pred[0, p0:p1].cpu(), color_dict["P"], lw=pred_lw)
    ax.plot(s_pred[0, p0:p1].cpu(), color_dict["S"], lw=pred_lw)

    row = generator.metadata.iloc[idx]
    if "phase_label" not in row:
        return

    if row["phase_label"] == "P":
        ax.axvline(np.argmax(p_pred[0, p0:p1].cpu()), c=color_dict["P"], lw=pred_lw)
    else:
        ax.axvline(np.argmax(s_pred[0, p0:p1].cpu()), c=color_dict["S"], lw=pred_lw)


def gpd_plot(idx, ax, model, generator):
    batch = generator[idx]
    x = torch.tensor(batch["X"]).unsqueeze(0)
    p0, p1 = batch["window_borders"]

    shape_save = x.shape
    x = x.reshape((-1,) + shape_save[2:])  # Merge batch and sliding window dimensions

    pred = model(x.to(model.device)).cpu()
    pred = pred.reshape(shape_save[:2] + (-1,))
    pred = torch.repeat_interleave(
        pred, model.predict_stride, dim=1
    )  # Counteract stride
    pred = F.pad(pred, (0, 0, 200, 200))
    pred = pred.permute(0, 2, 1)
    pred[
        :, 2, :200
    ] = 1  # Otherwise windows shorter 30 s will automatically produce detections
    pred[
        :, 2, -200:
    ] = 1  # Otherwise windows shorter 30 s will automatically produce detections

    # ax.plot(1 - pred[0, 2, p0:p1], color_dict["det"])
    ax.plot(pred[0, 0, p0:p1], color_dict["P"], lw=pred_lw)
    ax.plot(pred[0, 1, p0:p1], color_dict["S"], lw=pred_lw)

    row = generator.metadata.iloc[idx]
    if "phase_label" not in row:
        return

    phase_id = {"P": 0, "S": 1}[row["phase_label"]]

    ax.axvline(
        np.argmax(pred[0, phase_id, p0:p1]),
        c=color_dict[row["phase_label"]],
        lw=pred_lw,
    )


def phasenet_plot(idx, ax, model, generator):
    batch = generator[idx]
    x = torch.tensor(batch["X"]).unsqueeze(0)
    p0, p1 = batch["window_borders"]

    pred = model(x.to(model.device)).cpu()
    # ax.plot(1 - pred[0, 2, p0:p1])
    ax.plot(pred[0, 0, p0:p1], color_dict["P"], lw=pred_lw)
    ax.plot(pred[0, 1, p0:p1], color_dict["S"], lw=pred_lw)

    row = generator.metadata.iloc[idx]
    if "phase_label" not in row:
        return

    phase_id = {"P": 0, "S": 1}[row["phase_label"]]

    ax.axvline(
        np.argmax(pred[0, phase_id, p0:p1]),
        c=color_dict[row["phase_label"]],
        lw=pred_lw,
    )


def annotations_plot(idx, ax, axs, task_targets, data):
    row = task_targets.iloc[idx]
    data_idx = data.get_idx_from_trace_name(row["trace_name"])
    _, metadata = data.get_sample(data_idx)

    if "trace_type" in row:
        ax.text(
            0.01, 0.98, row["trace_type"], va="top", ha="left", transform=ax.transAxes
        )

    for col, phase in models.phase_dict.items():
        if col not in metadata:
            continue
        if np.isnan(metadata[col]):
            continue

        ax.axvline(metadata[col] - row["start_sample"], c=color_dict[phase], lw=pred_lw)

        for ax0 in axs:
            ax0.axvline(
                metadata[col] - row["start_sample"],
                c=color_dict[phase],
                ls="--",
                lw=0.7 * pred_lw,
            )


def load_models(basepath, source):
    model_dict = {}
    model_dict["basicphaseae"] = load_best_model(
        models.BasicPhaseAELit, basepath / f"{source}_basicphaseae", "version_0"
    )
    model_dict["cred"] = load_best_model(
        models.CREDLit, basepath / f"{source}_cred", "version_0"
    )
    model_dict["dpp_p"] = load_best_model(
        models.DPPPickerLit, basepath / f"{source}_dpppickerp", "version_0"
    )
    try:
        model_dict["dpp_s"] = load_best_model(
            models.DPPPickerLit, basepath / f"{source}_dpppickers", "version_0"
        )
    except FileNotFoundError:
        # Dummy to avoid exceptions later
        model_dict["dpp_s"] = model_dict["dpp_p"]
    model_dict["eqtransformer"] = load_best_model(
        models.EQTransformerLit, basepath / f"{source}_eqtransformer", "version_0"
    )
    model_dict["gpd"] = load_best_model(
        models.GPDLit, basepath / f"{source}_gpd", "version_0"
    )
    model_dict["gpdpick"] = load_best_model(
        models.GPDLit, basepath / f"{source}_gpdpick", "version_0"
    )
    model_dict["phasenet"] = load_best_model(
        models.PhaseNetLit, basepath / f"{source}_phasenet", "version_0"
    )

    # Move models to cuda
    for model in model_dict.values():
        model.cuda()
    return model_dict


def comparison_plot(
    source, target, data, idx=None, highpass=None, axs=None, legend=True
):
    # Load models
    basepath = Path("weights")

    model_dict = load_models(basepath, source)

    targetpath = Path("targets") / target / "task23.csv"

    task_targets = pd.read_csv(targetpath)
    task_targets = task_targets[task_targets["trace_split"] == "dev"]

    generator_dict = {}
    for model_name, model in model_dict.items():
        generator_dict[model_name] = sbg.SteeredGenerator(data, task_targets)
        generator_dict[model_name].add_augmentations(model.get_eval_augmentations())

    @ticker.FuncFormatter
    def time_formatter(x, pos):
        return f"{x / 100:.1f}"

    if idx is None:
        idx = np.random.randint(len(task_targets))

    if axs is None:
        fig = plt.figure(figsize=(15, 12))
        axs = fig.subplots(
            7,
            1,
            sharex=True,
            gridspec_kw={"hspace": 0.05, "height_ratios": [4, 1, 1, 1, 1, 1, 1]},
        )
    else:
        fig = None

    for ax in axs:
        ax.set_yticklabels([])

    for ax in axs[1:]:
        ax.set_ylim(0, 1.05)

    batch = generator_dict["eqtransformer"][idx]
    waveforms = batch["X"]
    if highpass is not None:
        sos = scipy.signal.butter(2, highpass, "hp", fs=100, output="sos")
        waveforms = scipy.signal.sosfiltfilt(sos, waveforms)

    p0, p1 = batch["window_borders"]
    waveforms = waveforms[:, p0:p1]
    handles = []
    for i, c in enumerate("ZNE"):
        handles.append(axs[0].plot(waveforms[i], label=c)[0])
    if legend:
        leg1 = axs[0].legend(handles=handles, loc="upper left")

    ph = axs[0].axvline(-100, c=color_dict["P"], label="P", lw=pred_lw)
    sh = axs[0].axvline(-100, c=color_dict["S"], label="S", lw=pred_lw)
    dh = axs[0].axvline(-100, c=color_dict["det"], label="Detection", lw=pred_lw)
    if legend:
        axs[0].legend(handles=[ph, sh, dh], loc="lower left")
        axs[0].add_artist(leg1)

    annotations_plot(idx, axs[0], axs, task_targets, data)
    with torch.no_grad():
        basicphaseae_plot(
            idx, axs[1], model_dict["basicphaseae"], generator_dict["basicphaseae"]
        )
        cred_plot(idx, axs[2], model_dict["cred"], generator_dict["cred"])
        dpp_plot(
            idx,
            axs[3],
            (model_dict["dpp_p"], model_dict["dpp_s"]),
            generator_dict["dpp_p"],
        )
        eqtransformer_plot(
            idx, axs[4], model_dict["eqtransformer"], generator_dict["eqtransformer"]
        )
        gpd_plot(idx, axs[5], model_dict["gpdpick"], generator_dict["gpdpick"])
        phasenet_plot(idx, axs[6], model_dict["phasenet"], generator_dict["phasenet"])

    axs[0].set_xlim(0, waveforms.shape[1])
    axs[-1].set_xlabel("Time [s]")
    axs[0].xaxis.set_major_formatter(time_formatter)
    axs[1].set_ylabel("BPAE")
    axs[2].set_ylabel("CRED")
    axs[3].set_ylabel("DPP")
    axs[4].set_ylabel("EQT")
    axs[5].set_ylabel("GPD")
    axs[6].set_ylabel("PN")

    axs[0].set_title(f"{source.upper()} to {target.upper()} idx={idx}")

    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)

    if not legend:
        # Return handles to allow plotting legend manually
        return fig, handles, [ph, sh, dh]
    else:
        return fig
