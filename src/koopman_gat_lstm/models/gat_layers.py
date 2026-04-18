"""Dense graph-attention helpers shared by Koopman variants."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def apply_source_logit_correction(
    logits: torch.Tensor,
    source_prior: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-head source-node bias before softmax attention normalization."""
    if logits.ndim != 4:
        raise ValueError("expected logits with shape [batch, heads, dst, src]")
    if source_prior.ndim != 2:
        raise ValueError("expected source_prior with shape [batch, src]")
    if beta.ndim != 1:
        raise ValueError("expected beta with shape [heads]")
    if logits.shape[0] != source_prior.shape[0]:
        raise ValueError("batch dimension does not match source prior")
    if logits.shape[1] != beta.shape[0]:
        raise ValueError("head dimension does not match beta")
    if logits.shape[-1] != source_prior.shape[-1]:
        raise ValueError("source dimension does not match source prior")

    gain = F.softplus(beta).view(1, -1, 1, 1)
    prior = source_prior.view(source_prior.shape[0], 1, 1, source_prior.shape[-1])
    return logits + gain * prior


def masked_attention_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize attention over valid sources while keeping fully masked rows at zero."""
    if logits.ndim != mask.ndim:
        raise ValueError("mask rank must match logits rank")

    try:
        expanded_mask = mask.expand_as(logits)
    except RuntimeError as exc:
        raise ValueError("mask shape must be broadcastable to logits shape") from exc

    row_max = logits.masked_fill(~expanded_mask, float("-inf")).max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    shifted_logits = (logits - row_max).masked_fill(~expanded_mask, float("-inf"))
    exp_logits = torch.exp(shifted_logits)
    denom = exp_logits.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)
    return exp_logits / denom


class DenseGraphAttentionLayer(nn.Module):
    """Small dense multi-head GAT layer for per-timestep node encoding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        *,
        concat: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat = concat

        self.weight = nn.Parameter(torch.empty(num_heads, input_dim, output_dim))
        self.attn_src = nn.Parameter(torch.empty(num_heads, output_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, output_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, output_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        *,
        source_prior: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected x with shape [batch, nodes, features]")
        if adj.ndim != 2:
            raise ValueError("expected adj with shape [nodes, nodes]")
        if adj.shape[0] != x.shape[1] or adj.shape[1] != x.shape[1]:
            raise ValueError("adjacency shape does not match node dimension")
        if (source_prior is None) != (beta is None):
            raise ValueError("source_prior and beta must be provided together")

        projected = torch.einsum("bni,hio->bhno", x, self.weight) + self.bias.view(1, self.num_heads, 1, self.output_dim)
        src_scores = torch.einsum("bhnd,hd->bhn", projected, self.attn_src)
        dst_scores = torch.einsum("bhnd,hd->bhn", projected, self.attn_dst)
        logits = self.leaky_relu(dst_scores.unsqueeze(-1) + src_scores.unsqueeze(-2))

        if source_prior is not None and beta is not None:
            logits = apply_source_logit_correction(logits, source_prior, beta)

        mask = adj.to(dtype=torch.bool, device=x.device).view(1, 1, adj.shape[0], adj.shape[1])
        attention = masked_attention_softmax(logits, mask)
        output = torch.einsum("bhij,bhjd->bhid", attention, projected)

        if self.concat:
            encoded = output.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.num_heads * self.output_dim)
        else:
            encoded = output.mean(dim=1)

        if return_attention:
            return encoded, attention
        return encoded
