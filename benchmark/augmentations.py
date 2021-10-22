"""
This file contains augmentations required for the models that are too specific to be merged into SeisBench.
"""

import numpy as np
import copy


class DuplicateEvent:
    """
    Adds a rescaled version of the event to the empty part of the trace after the event.
    Event position and empty space are determined from a detection.
    Detections can be generated for example with :py:class:`~seisbench.generate.labeling.DetectionLabeller`.

    This implementation is modelled after the `implementation for EQTransformer <https://github.com/smousavi05/EQTransformer/blob/98676017f971efbb6f4475f42e415c3868d00c03/EQTransformer/core/EqT_utils.py#L255>`_.

    .. warning::
        This augmentation does **not** modify the metadata, as representing multiple picks of
        the same type is currently not supported. Workflows should therefore always first generate
        labels from metadata and then pass the labels in the key `label_keys`. These keys are automatically
        adjusted by addition of the labels.

    .. warning::
        This implementation currently has strict shape requirements:

        - (1, samples) for detection
        - (channels, samples) for data
        - (labels, samples) for labels

    :param inv_scale: The scale factor is defined by as 1/u, where u is uniform.
                      `inv_scale` defines the minimum and maximum values for u.
                      Defaults to (1, 10), e.g., scaling by factor 1 to 1/10.
    :param detection_key: Key to read detection from.
                          If key is a tuple, detection will be read from the first key and written to the second one.
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key
                to read from and the second one the key to write to.
    :param label_keys: Keys for the label columns.
                       Labels of the original and duplicate events will be added and capped at 1.
                       Note that this will lead to invalid noise traces.
                       Value can either be a single key specification or a list of key specifications.
                       Each key specification is either a string, for identical input and output keys,
                       or as a tuple of two strings, input and output keys.
                       Defaults to None.
    """

    def __init__(
        self, inv_scale=(1, 10), detection_key="detections", key="X", label_keys=None
    ):
        if isinstance(detection_key, str):
            self.detection_key = (detection_key, detection_key)
        else:
            self.detection_key = detection_key

        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        # Single key
        if not isinstance(label_keys, list):
            if label_keys is None:
                label_keys = []
            else:
                label_keys = [label_keys]

        # Resolve identical input and output keys
        self.label_keys = []
        for key in label_keys:
            if isinstance(key, tuple):
                self.label_keys.append(key)
            else:
                self.label_keys.append((key, key))

        self.inv_scale = inv_scale

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        detection, _ = state_dict[self.detection_key[0]]
        detection_mask = detection[0] > 0.5

        if detection.shape[-1] != x.shape[-1]:
            raise ValueError("Number of samples in trace and detection disagree.")

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        if detection_mask.any():
            n_samples = x.shape[-1]
            event_samples = np.arange(n_samples)[detection_mask]
            event_start, event_end = np.min(event_samples), np.max(event_samples) + 1

            if event_end + 20 < n_samples:
                second_start = np.random.randint(event_end + 20, n_samples)
                scale = 1 / np.random.uniform(*self.inv_scale)

                if self.key[0] != self.key[1]:
                    # Avoid inplace modification if input and output keys differ
                    x = x.copy()

                space = min(event_end - event_start, n_samples - second_start)
                x[:, second_start : second_start + space] += (
                    x[:, event_start : event_start + space] * scale
                )

                shift = second_start - event_start

                for label_key in self.label_keys + [self.detection_key]:
                    y, metadata = state_dict[label_key[0]]
                    if y.shape[-1] != n_samples:
                        raise ValueError(
                            f"Number of samples disagree between trace and label key '{label_key[0]}'."
                        )

                    if label_key[0] != label_key[1]:
                        metadata = copy.deepcopy(metadata)
                        y = y.copy()

                    y[:, shift:] += y[:, :-shift]
                    y = np.minimum(y, 1)
                    state_dict[label_key[1]] = (y, metadata)
        else:
            # Copy entries
            for label_key in self.label_keys + [self.detection_key]:
                y, metadata = state_dict[label_key[0]]
                if label_key[0] != label_key[1]:
                    metadata = copy.deepcopy(metadata)
                    y = y.copy()
                state_dict[label_key[1]] = (y, metadata)

        state_dict[self.key[1]] = (x, metadata)
