"""Effective / nominal bpw accounting for W3 dense PTQ simulation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch.nn as nn

DENSE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def is_dense_proj(name: str) -> bool:
    """True for module names (…q_proj) or parameter names (…q_proj.weight)."""
    parts = name.split(".")
    leaf = parts[-1]
    if leaf in DENSE_SUFFIXES:
        return True
    if leaf in ("weight", "bias") and len(parts) >= 2 and parts[-2] in DENSE_SUFFIXES:
        return True
    return False


def is_kept_high_precision(name: str, module: nn.Module) -> bool:
    leaf = name.split(".")[-1]
    if leaf in ("embed_tokens", "wte", "wpe", "lm_head"):
        return True
    if "norm" in leaf.lower() or "ln" in leaf.lower():
        return True
    return False


def collect_param_groups(model: nn.Module) -> Dict[str, List[Tuple[str, int]]]:
    dense: List[Tuple[str, int]] = []
    high: List[Tuple[str, int]] = []
    other: List[Tuple[str, int]] = []
    for name, p in model.named_parameters():
        n = p.numel()
        # Only count dense *weights* (not biases) toward W3 dense pool
        parts = name.split(".")
        is_dense_w = (
            len(parts) >= 2
            and parts[-1] == "weight"
            and parts[-2] in DENSE_SUFFIXES
        )
        if is_dense_w:
            dense.append((name, n))
        elif any(k in name for k in ("embed", "lm_head", "norm", "ln_")) or name.endswith(".bias"):
            high.append((name, n))
        else:
            other.append((name, n))
    return {"dense": dense, "high": high, "other": other}


def estimate_bpw(
    model: nn.Module,
    bits_dense: int = 3,
    bits_high: float = 16.0,
    bits_other: float = 16.0,
    group_size: int = 128,
    scale_bits: float = 16.0,
) -> Dict[str, float]:
    """
    Effective bpw estimate:
    - dense weights stored at bits_dense + per-group scale overhead
    - embeddings / norms / lm_head / biases at bits_high
    - leftover linear/other at bits_other
    """
    groups = collect_param_groups(model)
    n_dense = sum(n for _, n in groups["dense"])
    n_high = sum(n for _, n in groups["high"])
    n_other = sum(n for _, n in groups["other"])
    n_total = n_dense + n_high + n_other

    n_groups = 0
    for _, n in groups["dense"]:
        n_groups += (n + group_size - 1) // group_size

    bits_total = (
        n_dense * bits_dense
        + n_groups * scale_bits
        + n_high * bits_high
        + n_other * bits_other
    )
    bpw_eff = bits_total / max(n_total, 1)
    return {
        "n_params_total": float(n_total),
        "n_params_dense": float(n_dense),
        "n_params_high": float(n_high),
        "n_params_other": float(n_other),
        "n_groups_dense": float(n_groups),
        "bits_dense_nominal": float(bits_dense),
        "bpw_effective_est": float(bpw_eff),
        "fraction_dense": float(n_dense / max(n_total, 1)),
    }


def list_dense_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    out: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and is_dense_proj(name):
            out.append((name, mod))
    return out
