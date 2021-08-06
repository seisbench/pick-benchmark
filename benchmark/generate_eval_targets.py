"""
This script generates evaluation targets for the following three tasks:

- Earthquake detection (Task 1): Given a 30~s window, does the window contain an earthquake signal?
- Phase identification (Task 2): Given a 10~s window containing exactly one phase onset, identify which phase type.
- Onset time determination (Task 3): Given a 10~s window containing exactly one phase onset, identify the onset time.

Each target for evaluation will consist of the following information:

- trace name (as in dataset)
- trace index (in dataset)
- split (as in dataset)
- sampling rate (at which all information is presented)
- start_sample
- end_sample
- trace_type (only task 1: earthquake/noise)
- phase_label (only task 2/3: P/S)
- full_phase_label (only task 2/3: phase label as in the dataset, might be Pn, Pg, etc.)
- phase_onset_sample (only task 2/3: onset sample of the phase relative to full trace)

It needs to be provided with a dataset and writes a folder with two CSV files, one for task 1 and one for tasks 2 and 3.
Each file will describe targets for train, dev and test, derived from the respective splits.

When using these tasks for evaluation, the models can make use of waveforms from the context, i.e.,
before/after the start and end samples. However, make sure this does not add further bias in the evaluation,
for example by always centring the windows on the picks using context.

.. warning::
    For comparability, it is strongly advised to use published evaluation targets, instead of generating new ones.

.. warning::
    This implementation is not optimized and loads the full waveform data for its computations.
    This will lead to very high memory usage, as the full dataset will be stored in memory.
"""
import seisbench.data as sbd

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from models import phase_dict


def main(dataset_name, output, tasks, sampling_rate, noise_before_events):
    np.random.seed(42)
    tasks = [str(i) in tasks.split(",") for i in range(1, 4)]

    if not any(tasks):
        raise ValueError(f"No task defined. Got tasks='{tasks}'.")

    dataset_args = {
        "sampling_rate": sampling_rate,
        "dimension_order": "NCW",
        "cache": "full",
    }

    try:
        # Check if dataset is available in SeisBench
        dataset = sbd.__getattribute__(dataset_name)(**dataset_args)
    except AttributeError:
        # Otherwise interpret data_in as path
        dataset = sbd.WaveformDataset(dataset_name, **dataset_args)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    if "split" in dataset.metadata.columns:
        dataset.filter(dataset["split"].isin(["dev", "test"]), inplace=True)

    dataset.preload_waveforms(pbar=True)

    if tasks[0]:
        generate_task1(dataset, output, sampling_rate, noise_before_events)
    if tasks[1] or tasks[2]:
        generate_task23(dataset, output, sampling_rate)


def generate_task1(dataset, output, sampling_rate, noise_before_events):
    np.random.seed(42)
    windowlen = 30 * sampling_rate  # 30 s windows
    labels = []

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        waveforms, metadata = dataset.get_sample(i)

        if "split" in metadata:
            trace_split = metadata["split"]
        else:
            trace_split = ""

        def checkphase(metadata, phase, phase_label, target_phase, npts):
            return (
                phase in metadata
                and phase_label == target_phase
                and not np.isnan(metadata[phase])
                and 0 <= metadata[phase] < npts
            )

        p_arrivals = [
            metadata[phase]
            for phase, phase_label in phase_dict.items()
            if checkphase(metadata, phase, phase_label, "P", waveforms.shape[-1])
        ]
        s_arrivals = [
            metadata[phase]
            for phase, phase_label in phase_dict.items()
            if checkphase(metadata, phase, phase_label, "S", waveforms.shape[-1])
        ]

        if len(p_arrivals) == 0 and len(s_arrivals) == 0:
            start_sample, end_sample = select_window_containing(
                waveforms.shape[-1], windowlen
            )
            sample = {
                "trace_name": metadata["trace_name"],
                "trace_idx": i,
                "trace_split": trace_split,
                "sampling_rate": sampling_rate,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "trace_type": "noise",
            }
            labels += [sample]

        else:
            first_arrival = min(p_arrivals + s_arrivals)

            start_sample, end_sample = select_window_containing(
                waveforms.shape[-1], windowlen, containing=first_arrival
            )
            if end_sample - start_sample <= windowlen:
                sample = {
                    "trace_name": metadata["trace_name"],
                    "trace_idx": i,
                    "trace_split": trace_split,
                    "sampling_rate": sampling_rate,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "trace_type": "earthquake",
                }
                labels += [sample]

            if noise_before_events and first_arrival > windowlen:
                start_sample, end_sample = select_window_containing(
                    min(waveforms.shape[-1], first_arrival), windowlen
                )
                if end_sample - start_sample <= windowlen:
                    sample = {
                        "trace_name": metadata["trace_name"],
                        "trace_idx": i,
                        "trace_split": trace_split,
                        "sampling_rate": sampling_rate,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "trace_type": "noise",
                    }
                    labels += [sample]

    labels = pd.DataFrame(labels)
    diff = labels["end_sample"] - labels["start_sample"]
    labels = labels[diff > 100]
    labels.to_csv(output / "task1.csv", index=False)


def generate_task23(dataset, output, sampling_rate):
    np.random.seed(42)
    windowlen = 10 * sampling_rate  # 30 s windows
    labels = []

    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        waveforms, metadata = dataset.get_sample(idx)

        if "split" in metadata:
            trace_split = metadata["split"]
        else:
            trace_split = ""

        def checkphase(metadata, phase, npts):
            return (
                phase in metadata
                and not np.isnan(metadata[phase])
                and 0 <= metadata[phase] < npts
            )

        # Example entry: (1031, "P", "Pg")
        arrivals = sorted(
            [
                (metadata[phase], phase_label, phase.split("_")[1])
                for phase, phase_label in phase_dict.items()
                if checkphase(metadata, phase, waveforms.shape[-1])
            ]
        )

        if len(arrivals) == 0:
            # Trace has no arrivals
            continue

        for i, (onset, phase, full_phase) in enumerate(arrivals):
            if i == 0:
                onset_before = 0
            else:
                onset_before = int(arrivals[i - 1][0]) + int(
                    0.5 * sampling_rate
                )  # 0.5 s minimum spacing

            if i == len(arrivals) - 1:
                onset_after = np.inf
            else:
                onset_after = int(arrivals[i + 1][0]) - int(
                    0.5 * sampling_rate
                )  # 0.5 s minimum spacing

            if (
                onset_after - onset_before < windowlen
                or onset_before > onset
                or onset_after < onset
            ):
                # Impossible to isolate pick
                continue

            else:
                onset_after = min(onset_after, waveforms.shape[-1])
                # Shift everything to a "virtual" start at onset_before
                start_sample, end_sample = select_window_containing(
                    onset_after - onset_before,
                    windowlen=windowlen,
                    containing=onset - onset_before,
                    bounds=(50, 50),
                )
                start_sample += onset_before
                end_sample += onset_before
                if end_sample - start_sample <= windowlen:
                    sample = {
                        "trace_name": metadata["trace_name"],
                        "trace_idx": idx,
                        "trace_split": trace_split,
                        "sampling_rate": sampling_rate,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "phase_label": phase,
                        "full_phase_label": full_phase,
                        "phase_onset": onset,
                    }

                    labels += [sample]

    labels = pd.DataFrame(labels)
    diff = labels["end_sample"] - labels["start_sample"]
    labels = labels[diff > 100]
    labels.to_csv(output / "task23.csv", index=False)


def select_window_containing(npts, windowlen, containing=None, bounds=(100, 100)):
    """
    Selects a window from a larger trace.

    :param npts: Number of points of the full trace
    :param windowlen: Desired windowlen
    :param containing: Sample number that should be contained. If None, any window within the trace is valid.
    :param bounds: The containing sample may not be in the first/last samples indicated here.
    :return: Start sample, end_sample
    """
    if npts <= windowlen:
        # If npts is smaller than the window length, always return the full window
        return 0, npts

    else:
        if containing is None:
            start_sample = np.random.randint(0, npts - windowlen + 1)
            return start_sample, start_sample + windowlen

        else:
            earliest_start = max(0, containing - windowlen + bounds[1])
            latest_start = min(npts - windowlen, containing - bounds[0])
            if latest_start <= earliest_start:
                # Again, return full window
                return 0, npts

            else:
                start_sample = np.random.randint(earliest_start, latest_start + 1)
                return start_sample, start_sample + windowlen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate evaluation targets. See the docstring for details."
    )
    parser.add_argument(
        "dataset", type=str, help="Path to input dataset or SeisBench dataset name"
    )
    parser.add_argument(
        "output", type=str, help="Path to write target files to. Must not exist."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="1,2,3",
        help="Which tasks to generate data for. By default generates data for all tasks.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=100,
        help="Sampling rate in Hz to generate targets for.",
    )
    parser.add_argument(
        "--no_noise_before_events",
        action="store_true",
        help="If set, does not extract noise from windows before the first arrival.",
    )
    args = parser.parse_args()

    main(
        args.dataset,
        args.output,
        args.tasks,
        args.sampling_rate,
        not args.no_noise_before_events,
    )
