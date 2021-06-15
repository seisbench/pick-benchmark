from benchmark.augmentations import DuplicateEvent

import numpy as np
import pytest
from unittest.mock import patch


def test_duplicate_event_keys():
    duplicate = DuplicateEvent(key="X")
    assert duplicate.key == ("X", "X")

    duplicate = DuplicateEvent(label_keys="X")
    assert duplicate.label_keys == [("X", "X")]

    duplicate = DuplicateEvent(label_keys=("X1", "X2"))
    assert duplicate.label_keys == [("X1", "X2")]

    duplicate = DuplicateEvent(label_keys=["X", ("y1", "y2")])
    assert duplicate.label_keys == [("X", "X"), ("y1", "y2")]


def test_duplicate_event_shape_errors():
    duplicate = DuplicateEvent(label_keys=("y"))

    state_dict = {
        "X": (np.random.rand(3, 1000), {}),
        "y": (np.random.rand(2, 1000), {}),
        "detections": (np.zeros((1, 1000)), {}),
    }
    duplicate(state_dict)

    state_dict = {
        "X": (np.random.rand(3, 1000), {}),
        "y": (np.random.rand(2, 1000), {}),
        "detections": (np.zeros((1, 1001)), {}),
    }
    with pytest.raises(ValueError):
        duplicate(state_dict)

    detections = np.zeros((1, 1000))
    detections[0, 100:200] = 1
    state_dict = {
        "X": (np.random.rand(3, 1000), {}),
        "y": (np.random.rand(2, 1010), {}),
        "detections": (detections, {}),
    }
    with pytest.raises(ValueError):
        duplicate(state_dict)


def test_duplicate_event():
    duplicate = DuplicateEvent(
        key=("X", "X2"),
        label_keys=("y", "y2"),
        detection_key=("detections", "detections2"),
    )

    # No detection, data unmodified
    detections = np.zeros((1, 1000))
    state_dict = {
        "X": (np.random.rand(3, 1000), {}),
        "y": (np.random.rand(2, 1000), {}),
        "detections": (detections, {}),
    }
    duplicate(state_dict)
    assert (state_dict["X"][0] == state_dict["X2"][0]).all()
    assert (state_dict["y"][0] == state_dict["y2"][0]).all()
    assert (state_dict["detections"][0] == state_dict["detections2"][0]).all()

    # No detection, data unmodified
    detections = np.zeros((1, 1000))
    detections[:, 100:200] = 1
    state_dict = {
        "X": (np.random.rand(3, 1000), {}),
        "y": (np.random.rand(2, 1000), {}),
        "detections": (detections, {}),
    }

    with patch("numpy.random.randint") as randint:
        with patch("numpy.random.uniform") as uniform:
            uniform.return_value = 2
            randint.return_value = 300
            duplicate(state_dict)

    target_x = state_dict["X"][0].copy()
    target_x[:, 300:400] += 0.5 * target_x[:, 100:200]

    target_y = state_dict["y"][0].copy()
    target_y[:, 200:] += target_y[:, :-200]
    target_y = np.minimum(target_y, 1)

    target_d = state_dict["detections"][0].copy()
    target_d[:, 200:] += target_d[:, :-200]

    assert np.allclose(target_x, state_dict["X2"][0])
    assert np.allclose(target_y, state_dict["y2"][0])
    assert np.allclose(target_d, state_dict["detections2"][0])
