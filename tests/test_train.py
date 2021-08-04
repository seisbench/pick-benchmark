import seisbench.data as sbd

from benchmark.train import apply_training_fraction


def test_apply_training_fraction():
    data = sbd.DummyDataset()
    unique_blocks_org = data["trace_name"].apply(lambda x: x.split("$")[0]).unique()

    apply_training_fraction(0.5, data)

    unique_blocks_new = data["trace_name"].apply(lambda x: x.split("$")[0]).unique()

    assert len(unique_blocks_new) == int(0.5 * len(unique_blocks_org))
