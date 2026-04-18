from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PREDICTIONS_FILE = Path("predictions") / "test_predictions.npz"


def _load_predictions(run_dir: Path) -> dict[str, np.ndarray]:
    predictions_path = run_dir / _PREDICTIONS_FILE
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    with np.load(predictions_path, allow_pickle=False) as predictions:
        required_keys = {"y_pred", "y_true", "case_ids"}
        missing_keys = sorted(required_keys.difference(predictions.files))
        if missing_keys:
            raise ValueError(f"predictions file is missing required keys: {', '.join(missing_keys)}")

        y_pred = np.asarray(predictions["y_pred"])
        y_true = np.asarray(predictions["y_true"])
        case_ids = np.asarray(predictions["case_ids"], dtype=str)

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have identical shapes within each run")

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "case_ids": case_ids,
    }


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def _compute_per_node_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> list[dict[str, float]]:
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have identical shapes within each run")
    if y_pred.ndim < 2:
        raise ValueError("prediction tensors must include a node dimension")

    node_metrics: list[dict[str, float]] = []
    for node_index in range(y_pred.shape[-1]):
        node_pred = y_pred[..., node_index]
        node_true = y_true[..., node_index]
        metrics = _compute_metrics(node_pred, node_true)
        node_metrics.append({"node": node_index + 1, **metrics})
    return node_metrics


def _load_entropy_curve(run_dir: Path, selected_case_id: str, layer_name: str) -> np.ndarray:
    entropy_path = run_dir / "cases" / selected_case_id / f"{layer_name}_entropy.npy"
    if not entropy_path.is_file():
        raise FileNotFoundError(entropy_path)

    entropy = np.asarray(np.load(entropy_path, allow_pickle=False))
    if entropy.ndim != 1:
        raise ValueError(f"{layer_name} entropy curve must be one-dimensional")
    return entropy


def _save_entropy_comparison_plot(
    comparison_dir: Path,
    layer_name: str,
    time_s: np.ndarray,
    koopman_entropy: np.ndarray,
    standard_entropy: np.ndarray,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_s, koopman_entropy, label="koopman", linewidth=1.5)
    ax.plot(time_s, standard_entropy, label="standard", linewidth=1.5, linestyle="--")
    ax.set_xlabel("time_s")
    ax.set_ylabel(f"{layer_name}_entropy")
    ax.set_title(f"attention_entropy_comparison_{layer_name}")
    ax.legend()
    fig.tight_layout()

    plot_path = comparison_dir / f"attention_entropy_comparison_{layer_name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def write_formal_comparison_artifacts(
    *,
    koopman_run_dir: Path,
    standard_run_dir: Path,
    comparison_dir: Path,
    selected_case_id: str,
    key_nodes: Sequence[int],
    input_start_s: float = 5.0,
    sample_rate_hz: int = 100,
) -> dict[str, Any]:
    koopman_predictions = _load_predictions(Path(koopman_run_dir))
    standard_predictions = _load_predictions(Path(standard_run_dir))

    if koopman_predictions["y_pred"].shape != standard_predictions["y_pred"].shape:
        raise ValueError("y_pred shapes must match across model runs")
    if koopman_predictions["y_true"].shape != standard_predictions["y_true"].shape:
        raise ValueError("y_true shapes must match across model runs")

    koopman_case_ids = [str(case_id) for case_id in koopman_predictions["case_ids"]]
    standard_case_ids = [str(case_id) for case_id in standard_predictions["case_ids"]]
    if koopman_case_ids != standard_case_ids:
        raise ValueError("case ID order must match across model runs")
    if selected_case_id not in koopman_case_ids:
        raise ValueError(f"selected case was not found in prediction case IDs: {selected_case_id}")

    if not np.array_equal(koopman_predictions["y_true"], standard_predictions["y_true"]):
        raise ValueError("target tensors must be numerically equal across model runs")

    koopman_layer1_entropy = _load_entropy_curve(Path(koopman_run_dir), selected_case_id, "layer1")
    standard_layer1_entropy = _load_entropy_curve(Path(standard_run_dir), selected_case_id, "layer1")
    koopman_layer2_entropy = _load_entropy_curve(Path(koopman_run_dir), selected_case_id, "layer2")
    standard_layer2_entropy = _load_entropy_curve(Path(standard_run_dir), selected_case_id, "layer2")

    if not (
        koopman_layer1_entropy.shape
        == standard_layer1_entropy.shape
        == koopman_layer2_entropy.shape
        == standard_layer2_entropy.shape
    ):
        raise ValueError("entropy curve shapes must match across model runs and layers")

    koopman_overall = _compute_metrics(koopman_predictions["y_pred"], koopman_predictions["y_true"])
    standard_overall = _compute_metrics(standard_predictions["y_pred"], standard_predictions["y_true"])

    koopman_node_metrics = _compute_per_node_metrics(
        koopman_predictions["y_pred"], koopman_predictions["y_true"]
    )
    standard_node_metrics = _compute_per_node_metrics(
        standard_predictions["y_pred"], standard_predictions["y_true"]
    )

    if len(koopman_node_metrics) != len(standard_node_metrics):
        raise ValueError("prediction node counts must match across model runs")

    available_nodes = len(koopman_node_metrics)
    nodes = [int(node) for node in key_nodes]
    if not nodes:
        raise ValueError("key_nodes must contain at least one node")
    if any(node < 1 for node in nodes):
        raise ValueError("key node numbers must be positive 1-based BUS numbers")
    if any(node > available_nodes for node in nodes):
        raise ValueError("requested key node exceeds available node columns")

    comparison_dir = Path(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics_path = comparison_dir / "overall_metrics_comparison.csv"
    pd.DataFrame(
        [
            {"model": "koopman", **koopman_overall},
            {"model": "standard", **standard_overall},
        ]
    ).to_csv(overall_metrics_path, index=False)

    key_node_rows: list[dict[str, Any]] = []
    for node in nodes:
        koopman_metrics = koopman_node_metrics[node - 1]
        standard_metrics = standard_node_metrics[node - 1]
        key_node_rows.append(
            {
                "node": node,
                "koopman_mse": koopman_metrics["mse"],
                "standard_mse": standard_metrics["mse"],
                "koopman_rmse": koopman_metrics["rmse"],
                "standard_rmse": standard_metrics["rmse"],
                "koopman_mae": koopman_metrics["mae"],
                "standard_mae": standard_metrics["mae"],
                "rmse_delta": koopman_metrics["rmse"] - standard_metrics["rmse"],
                "mae_delta": koopman_metrics["mae"] - standard_metrics["mae"],
            }
        )

    key_node_metrics_path = comparison_dir / "key_node_metrics_comparison.csv"
    pd.DataFrame(key_node_rows).to_csv(key_node_metrics_path, index=False)

    time_step = np.arange(koopman_layer1_entropy.shape[0], dtype=int)
    time_s = input_start_s + time_step / float(sample_rate_hz)
    entropy_frame = pd.DataFrame(
        {
            "time_step": time_step,
            "time_s": time_s,
            "koopman_layer1_entropy": koopman_layer1_entropy,
            "standard_layer1_entropy": standard_layer1_entropy,
            "koopman_layer2_entropy": koopman_layer2_entropy,
            "standard_layer2_entropy": standard_layer2_entropy,
        }
    )
    attention_entropy_path = comparison_dir / "attention_entropy_comparison.csv"
    entropy_frame.to_csv(attention_entropy_path, index=False)

    layer1_plot_path = _save_entropy_comparison_plot(
        comparison_dir,
        "layer1",
        time_s,
        koopman_layer1_entropy,
        standard_layer1_entropy,
    )
    layer2_plot_path = _save_entropy_comparison_plot(
        comparison_dir,
        "layer2",
        time_s,
        koopman_layer2_entropy,
        standard_layer2_entropy,
    )

    return {
        "selected_case_id": selected_case_id,
        "overall_metrics": {
            "koopman": koopman_overall,
            "standard": standard_overall,
        },
        "paths": {
            "overall_metrics_comparison": str(overall_metrics_path),
            "key_node_metrics_comparison": str(key_node_metrics_path),
            "attention_entropy_comparison": str(attention_entropy_path),
            "attention_entropy_comparison_layer1": str(layer1_plot_path),
            "attention_entropy_comparison_layer2": str(layer2_plot_path),
        },
    }
