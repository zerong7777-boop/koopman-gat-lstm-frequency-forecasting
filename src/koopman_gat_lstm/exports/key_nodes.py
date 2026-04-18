from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, key_nodes: Iterable[int]) -> list[int]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays shaped [forecast_steps, nodes]")

    nodes = [int(node) for node in key_nodes]
    if not nodes:
        raise ValueError("key_nodes must contain at least one node")

    if any(node < 1 for node in nodes):
        raise ValueError("key node numbers must be positive 1-based BUS numbers")
    if any(node > y_true.shape[1] for node in nodes):
        raise ValueError("requested key node exceeds available node columns")

    return nodes


def export_key_node_comparisons(
    case_dir: Path | str,
    case_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    key_nodes: Iterable[int],
) -> dict[str, object]:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    nodes = _validate_inputs(y_true, y_pred, key_nodes)
    time_s = 5.40 + np.arange(y_true.shape[0]) * 0.01

    excel_path = case_dir / f"{case_id}_key_node_comparison.xlsx"
    plots: dict[int, Path] = {}

    with pd.ExcelWriter(excel_path) as writer:
        for node in nodes:
            column_index = node - 1
            sheet_name = f"node_{node}"
            frame = pd.DataFrame(
                {
                    "time_s": time_s,
                    "true": y_true[:, column_index],
                    "pred": y_pred[:, column_index],
                }
            )
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_s, y_true[:, column_index], label="true", linewidth=1.5)
            ax.plot(time_s, y_pred[:, column_index], label="pred", linestyle="--", linewidth=1.5)
            ax.set_xlabel("time_s")
            ax.set_ylabel(f"node_{node}")
            ax.set_title(f"{case_id} node {node}")
            ax.legend()
            fig.tight_layout()

            plot_path = case_dir / f"node_{node}_prediction.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots[node] = plot_path

    return {"excel": excel_path, "plots": plots}
