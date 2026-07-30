"""Group-wise 3-bit symmetric quantization (RTN and MSE scale search)."""

from __future__ import annotations

from typing import Literal, Tuple

import torch

BITS = 3
N_LEVELS = (1 << BITS) - 1  # 7 levels for symmetric: -3..+3 with zero
QMAX = (1 << (BITS - 1)) - 1  # 3


def _group_view(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int, int]:
    """Flatten last-dim groups. Returns (groups[G, gs], out_features, in_features)."""
    assert w.ndim == 2
    out_f, in_f = w.shape
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (group_size - (n % group_size)) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    groups = flat.view(-1, group_size)
    return groups, out_f, in_f


def _degroup(groups: torch.Tensor, out_f: int, in_f: int) -> torch.Tensor:
    n = out_f * in_f
    return groups.reshape(-1)[:n].view(out_f, in_f)


def rtn_quantize(
    w: torch.Tensor,
    group_size: int = 128,
    scale_mode: Literal["maxabs", "absmean"] = "maxabs",
    scale_mult: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Symmetric per-group RTN to 3-bit grid. Returns dequantized float weights + meta."""
    orig_dtype = w.dtype
    w32 = w.detach().float()
    groups, out_f, in_f = _group_view(w32, group_size)
    if scale_mode == "maxabs":
        scales = groups.abs().amax(dim=1).clamp(min=1e-12)
    else:
        scales = groups.abs().mean(dim=1).clamp(min=1e-12) * 2.0  # heuristic span
    scales = scales * float(scale_mult)
    # map to [-QMAX, QMAX]
    q = torch.round(groups / scales.unsqueeze(1) * QMAX).clamp(-QMAX, QMAX)
    deq = q / QMAX * scales.unsqueeze(1)
    out = _degroup(deq, out_f, in_f).to(orig_dtype)
    meta = {
        "bits": BITS,
        "group_size": group_size,
        "scale_mode": scale_mode,
        "scale_mult": scale_mult,
        "n_groups": int(groups.shape[0]),
        "symmetric": True,
    }
    return out, meta


def mse_quantize(
    w: torch.Tensor,
    group_size: int = 128,
    scale_mode: Literal["maxabs", "absmean"] = "maxabs",
    grid: Tuple[float, ...] = (
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
        1.0,
        1.05,
        1.15,
        1.25,
        1.4,
        1.6,
    ),
) -> Tuple[torch.Tensor, dict]:
    """RTN with per-tensor (weight) MSE-optimal scale multiplier grid search."""
    orig_dtype = w.dtype
    w32 = w.detach().float()
    best_mse = float("inf")
    best_out = None
    best_mult = 1.0
    for m in grid:
        cand, meta = rtn_quantize(w32, group_size=group_size, scale_mode=scale_mode, scale_mult=m)
        mse = torch.mean((cand - w32) ** 2).item()
        if mse < best_mse:
            best_mse = mse
            best_out = cand
            best_mult = m
    assert best_out is not None
    meta = {
        "bits": BITS,
        "group_size": group_size,
        "scale_mode": scale_mode,
        "scale_mult": best_mult,
        "best_mse": best_mse,
        "symmetric": True,
        "clip": "mse_grid",
    }
    return best_out.to(orig_dtype), meta


def quantize_weight(
    w: torch.Tensor,
    method: Literal["rtn", "mse"],
    group_size: int = 128,
) -> Tuple[torch.Tensor, dict]:
    if method == "rtn":
        return rtn_quantize(w, group_size=group_size)
    if method == "mse":
        return mse_quantize(w, group_size=group_size)
    raise ValueError(f"unknown method {method}")
