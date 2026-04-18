import numpy as np

from koopman_gat_lstm.data.artifacts import (
    MaterializedDataset,
    load_dataset_artifact,
    save_dataset_artifact,
)
from koopman_gat_lstm.data.splits import split_case_ids


def test_split_case_ids_is_deterministic_and_covers_all_cases():
    split = split_case_ids([f"case-{i}" for i in range(10)], train=0.7, val=0.1, test=0.2, seed=42)

    assert len(split["train"]) == 7
    assert len(split["val"]) == 1
    assert len(split["test"]) == 2
    assert sorted(split["train"] + split["val"] + split["test"]) == [f"case-{i}" for i in range(10)]


def test_split_case_ids_respects_zero_fraction_splits():
    split = split_case_ids(["a", "b", "c"], train=0.8, val=0.2, test=0.0, seed=42)

    assert len(split["train"]) == 2
    assert len(split["val"]) == 1
    assert split["test"] == []
    assert sorted(split["train"] + split["val"] + split["test"]) == ["a", "b", "c"]


def test_dataset_artifact_round_trips_materialized_arrays(tmp_path):
    artifact = MaterializedDataset(
        case_ids=np.array(["case-a", "case-b"]),
        x=np.arange(12, dtype=float).reshape(2, 2, 3),
        y=np.arange(18, dtype=float).reshape(2, 3, 3),
        koopman=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        adjacency=np.eye(3),
        frequency_mean=np.array([10.0, 20.0, 30.0]),
        frequency_std=np.array([1.0, 2.0, 3.0]),
        split={"train": ["case-a"], "val": [], "test": ["case-b"]},
    )

    artifact_path = save_dataset_artifact(tmp_path, artifact)
    loaded = load_dataset_artifact(artifact_path)

    assert artifact_path.name == "dataset.npz"
    assert np.array_equal(loaded.case_ids, artifact.case_ids)
    assert np.array_equal(loaded.x, artifact.x)
    assert np.array_equal(loaded.y, artifact.y)
    assert np.array_equal(loaded.koopman, artifact.koopman)
    assert np.array_equal(loaded.adjacency, artifact.adjacency)
    assert np.array_equal(loaded.frequency_mean, artifact.frequency_mean)
    assert np.array_equal(loaded.frequency_std, artifact.frequency_std)
    assert loaded.split == artifact.split
