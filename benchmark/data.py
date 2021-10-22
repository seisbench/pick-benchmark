"""
This file contains functionality related to data.
"""

import seisbench.data as sbd


def get_dataset_by_name(name):
    """
    Resolve dataset name to class from seisbench.data.

    :param name: Name of dataset as defined in seisbench.data.
    :return: Dataset class from seisbench.data
    """
    try:
        return sbd.__getattribute__(name)
    except AttributeError:
        raise ValueError(f"Unknown dataset '{name}'.")
