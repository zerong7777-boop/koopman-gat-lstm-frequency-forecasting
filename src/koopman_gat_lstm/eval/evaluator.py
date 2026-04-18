from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from koopman_gat_lstm.eval.attention import summarize_layer_attention
from koopman_gat_lstm.eval.metrics import compute_metrics
from koopman_gat_lstm.exports.case_exports import build_case_dir


def _batch_case_ids(batch, case_ids: Sequence[str], offset: int, batch_size: int) -> list[str]:
    if isinstance(batch, Mapping) and "case_id" in batch:
        raw_case_ids = batch["case_id"]
        if isinstance(raw_case_ids, str):
            return [raw_case_ids]
        return [str(case_id) for case_id in raw_case_ids]
    return [str(case_id) for case_id in case_ids[offset : offset + batch_size]]


def _unpack_batch(
    batch,
    case_ids: Sequence[str],
    offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[str]]:
    if isinstance(batch, Mapping):
        x = batch["x"]
        y = batch["y"]
        koopman = batch.get("koopman")
    else:
        x, y = batch[0], batch[1]
        koopman = batch[2] if len(batch) > 2 and torch.is_tensor(batch[2]) else None

    x = x.to(device)
    y = y.to(device)
    if koopman is not None:
        koopman = koopman.to(device)
    batch_case_ids = _batch_case_ids(batch, case_ids, offset, x.shape[0])
    if len(batch_case_ids) != x.shape[0]:
        raise ValueError(
            f"case ID count {len(batch_case_ids)} does not match batch size {x.shape[0]}"
        )
    return x, y, koopman, batch_case_ids


def _adjacency_to_device(adjacency, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(adjacency):
        return adjacency.to(device)
    return torch.as_tensor(adjacency, dtype=torch.float32, device=device)


def _forward_with_attention(
    model: torch.nn.Module,
    x: torch.Tensor,
    adjacency: torch.Tensor,
    koopman: torch.Tensor | None,
    uses_koopman: bool,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    if uses_koopman:
        if koopman is None:
            raise ValueError("koopman batch tensor is required when uses_koopman=True")
        return model(x, adjacency, koopman, return_attention=True)
    return model(x, adjacency, return_attention=True)


def evaluate_model(
    model,
    loader,
    adjacency,
    run_dir,
    case_ids,
    uses_koopman,
    device,
    selected_case_id=None,
) -> dict:
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    adjacency = _adjacency_to_device(adjacency, device)

    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    y_pred_batches: list[torch.Tensor] = []
    y_true_batches: list[torch.Tensor] = []
    ordered_case_ids: list[str] = []
    selected_attention: dict[str, torch.Tensor] | None = None
    selected_case_id = str(selected_case_id) if selected_case_id is not None else None
    case_ids = [str(case_id) for case_id in case_ids]
    offset = 0

    with torch.no_grad():
        for batch in loader:
            x, y, koopman, batch_case_ids = _unpack_batch(batch, case_ids, offset, device)
            if selected_case_id is None and batch_case_ids:
                selected_case_id = batch_case_ids[0]

            y_pred, attention = _forward_with_attention(model, x, adjacency, koopman, uses_koopman)
            y_pred_batches.append(y_pred.detach().cpu())
            y_true_batches.append(y.detach().cpu())
            ordered_case_ids.extend(batch_case_ids)

            if selected_attention is None and selected_case_id in batch_case_ids:
                selected_index = batch_case_ids.index(selected_case_id)
                selected_attention = {
                    layer_name: layer_attention[selected_index].detach().cpu()
                    for layer_name, layer_attention in attention.items()
                }
            offset += x.shape[0]

    if not y_pred_batches:
        raise ValueError("loader must contain at least one batch")
    if selected_case_id is None:
        raise ValueError("selected_case_id could not be inferred from case_ids or loader")
    if selected_attention is None:
        raise ValueError(f"selected case id was not evaluated: {selected_case_id}")

    y_pred_all = torch.cat(y_pred_batches, dim=0)
    y_true_all = torch.cat(y_true_batches, dim=0)
    metrics = compute_metrics(y_pred_all, y_true_all)

    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez(
        predictions_dir / "test_predictions.npz",
        y_pred=y_pred_all.numpy(),
        y_true=y_true_all.numpy(),
        case_ids=np.asarray(ordered_case_ids, dtype=str),
    )

    case_dir = build_case_dir(run_dir, selected_case_id)
    for layer_name in ("layer1", "layer2"):
        np.save(
            case_dir / f"{layer_name}_entropy.npy",
            summarize_layer_attention(selected_attention[layer_name])["curve"],
        )

    return {
        "metrics": metrics,
        "case_ids": ordered_case_ids,
        "selected_case_id": selected_case_id,
    }
