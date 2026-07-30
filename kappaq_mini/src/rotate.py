"""Orthogonal rotation helpers for PTQ arms."""

from __future__ import annotations

from typing import Optional

import torch


def make_orthogonal(dim: int, seed: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """QR-orthogonal matrix of shape [dim, dim], seeded for reproducibility."""
    device = device or torch.device("cpu")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    a = torch.randn(dim, dim, generator=g, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    # Fix sign ambiguity for determinism
    diag = torch.diagonal(r)
    signs = torch.sign(diag)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(device)


def rotate_weight(w: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply input-side rotation: W' = W @ R  (in_features must match R)."""
    assert w.ndim == 2
    assert R.shape[0] == w.shape[1] == R.shape[1]
    return (w.float() @ R.to(dtype=torch.float32, device=w.device)).to(w.dtype)


def unrotate_weight(w_rot_q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Undo rotation after quant: W_hat = W'_q @ R^T  so effective forward uses W_hat @ x ≈ W @ R @ R^T @ x."""
    assert w_rot_q.ndim == 2
    Rt = R.T.contiguous()
    return (w_rot_q.float() @ Rt.to(dtype=torch.float32, device=w_rot_q.device)).to(w_rot_q.dtype)
