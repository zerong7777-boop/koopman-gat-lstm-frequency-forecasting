from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch


def select_best_checkpoint_metric(
    val_metrics: Mapping[str, float],
    metric_name: str = "rmse",
) -> float:
    """Return the validation metric used for best-checkpoint selection."""
    if metric_name not in val_metrics:
        raise KeyError(metric_name)
    return float(val_metrics[metric_name])


@dataclass
class TrainerDefaults:
    """Small holder for training defaults until the full trainer exists."""

    loss_fn: torch.nn.Module = field(default_factory=torch.nn.MSELoss)
    best_checkpoint_metric_name: str = "rmse"

    def best_checkpoint_metric(self, val_metrics: Mapping[str, float]) -> float:
        return select_best_checkpoint_metric(val_metrics, self.best_checkpoint_metric_name)


def _unpack_batch(
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, Mapping):
        x = batch["x"]
        y = batch["y"]
        koopman = batch.get("koopman")
    else:
        x, y = batch[0], batch[1]
        koopman = batch[2] if len(batch) > 2 else None

    x = x.to(device)
    y = y.to(device)
    if koopman is not None:
        koopman = koopman.to(device)
    return x, y, koopman


def _forward_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    adjacency: torch.Tensor,
    koopman: torch.Tensor | None,
    uses_koopman: bool,
) -> torch.Tensor:
    if uses_koopman:
        if koopman is None:
            raise ValueError("koopman batch tensor is required when uses_koopman=True")
        return model(x, adjacency, koopman)
    return model(x, adjacency)


def _raise_for_prediction_target_shape_mismatch(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    stage: str,
) -> None:
    if y_pred.shape != y.shape:
        raise ValueError(
            f"{stage} prediction shape {tuple(y_pred.shape)} does not match "
            f"target shape {tuple(y.shape)}"
        )


def train_model(
    model,
    train_loader,
    val_loader,
    adjacency,
    run_dir,
    max_epochs,
    learning_rate,
    patience,
    device,
    uses_koopman,
) -> dict:
    device = torch.device(device)
    model = model.to(device)
    adjacency = adjacency.to(device)
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    best_val_rmse = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_element_count = 0
        for batch in train_loader:
            x, y, koopman = _unpack_batch(batch, device)

            optimizer.zero_grad()
            y_pred = _forward_model(model, x, adjacency, koopman, uses_koopman)
            _raise_for_prediction_target_shape_mismatch(y_pred, y, "train")
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1
            train_element_count += y.numel()

        if train_element_count == 0:
            raise ValueError("train_loader must contain at least one target element")

        model.eval()
        val_squared_error_sum = 0.0
        val_element_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, koopman = _unpack_batch(batch, device)
                y_pred = _forward_model(model, x, adjacency, koopman, uses_koopman)
                _raise_for_prediction_target_shape_mismatch(y_pred, y, "validation")
                error = y_pred - y
                val_squared_error_sum += float(error.pow(2).sum().item())
                val_element_count += y.numel()

        if val_element_count == 0:
            raise ValueError("val_loader must contain at least one target element")

        train_loss = train_loss_sum / train_batches
        val_rmse = (val_squared_error_sum / val_element_count) ** 0.5
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_rmse": float(val_rmse),
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = float(val_rmse)
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": best_val_rmse,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return {"best_val_rmse": float(best_val_rmse), "history": history}
