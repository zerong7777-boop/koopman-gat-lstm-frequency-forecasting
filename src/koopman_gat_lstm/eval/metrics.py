from __future__ import annotations

from collections.abc import Mapping

import torch


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
    """Compute basic regression metrics for a prediction batch."""
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have identical shapes")

    error = y_pred - y_true
    mse = torch.mean(error.pow(2))
    rmse = torch.sqrt(mse)
    mae = torch.mean(error.abs())
    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
    }
