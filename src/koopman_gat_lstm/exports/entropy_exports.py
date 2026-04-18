from __future__ import annotations

from typing import Any

import numpy as np

from koopman_gat_lstm.eval.attention import summarize_layer_attention


def compute_layer_entropy_curve(attention: Any, eps: float = 1e-12) -> np.ndarray:
    """Collapse layer attention into a head-averaged entropy curve."""
    return summarize_layer_attention(attention, eps=eps)["curve"]
