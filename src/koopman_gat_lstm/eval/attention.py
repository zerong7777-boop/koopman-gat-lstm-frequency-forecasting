from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy_attention(attention: Any) -> np.ndarray:
    if hasattr(attention, "detach"):
        attention = attention.detach()
    if hasattr(attention, "cpu"):
        attention = attention.cpu()
    return np.asarray(attention, dtype=np.float64)


def summarize_layer_attention(attention: Any, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """Summarize layer-wise attention into per-head and head-averaged entropy values."""
    probs = _as_numpy_attention(attention)
    if probs.ndim < 4:
        raise ValueError("expected attention with shape [..., heads, nodes, neighbors]")

    positive = probs > 0.0
    entropy_terms = np.zeros_like(probs, dtype=np.float64)
    entropy_terms[positive] = probs[positive] * np.log(np.clip(probs[positive], eps, 1.0))
    entropy = -entropy_terms.sum(axis=-1)
    head_entropy = entropy.mean(axis=-1)
    curve = head_entropy.mean(axis=-1)
    if curve.ndim > 1:
        curve = curve.mean(axis=tuple(range(curve.ndim - 1)))
    return {
        "head_entropy": head_entropy,
        "curve": curve,
    }
