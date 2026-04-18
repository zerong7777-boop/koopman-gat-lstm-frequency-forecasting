from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MaterializedDataset:
    case_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    koopman: np.ndarray
    adjacency: np.ndarray
    frequency_mean: np.ndarray
    frequency_std: np.ndarray
    split: dict[str, list[str]]


def save_dataset_artifact(output_dir, artifact: MaterializedDataset) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "dataset.npz"
    np.savez_compressed(
        artifact_path,
        case_ids=artifact.case_ids,
        x=artifact.x,
        y=artifact.y,
        koopman=artifact.koopman,
        adjacency=artifact.adjacency,
        frequency_mean=artifact.frequency_mean,
        frequency_std=artifact.frequency_std,
        split=np.array(json.dumps(artifact.split, ensure_ascii=False)),
    )
    return artifact_path


def load_dataset_artifact(path) -> MaterializedDataset:
    with np.load(path, allow_pickle=False) as data:
        split = json.loads(str(data["split"].item()))
        return MaterializedDataset(
            case_ids=data["case_ids"],
            x=data["x"],
            y=data["y"],
            koopman=data["koopman"],
            adjacency=data["adjacency"],
            frequency_mean=data["frequency_mean"],
            frequency_std=data["frequency_std"],
            split=split,
        )
