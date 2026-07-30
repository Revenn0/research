#!/usr/bin/env python3
"""Unit tests for quant roundtrip shapes and rotation orthogonality."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant import mse_quantize, rtn_quantize  # noqa: E402
from rotate import make_orthogonal, rotate_weight, unrotate_weight  # noqa: E402


def test_rtn_shapes():
    for shape in [(64, 128), (128, 256), (100, 100), (7, 13)]:
        w = torch.randn(*shape)
        q, meta = rtn_quantize(w, group_size=128)
        assert q.shape == w.shape, (q.shape, w.shape)
        assert meta["bits"] == 3
        assert torch.isfinite(q).all()
    print("PASS test_rtn_shapes")


def test_mse_improves_or_equals_rtn_mse():
    w = torch.randn(128, 256)
    q_rtn, _ = rtn_quantize(w, group_size=128)
    q_mse, meta = mse_quantize(w, group_size=128)
    mse_rtn = torch.mean((q_rtn - w) ** 2).item()
    mse_mse = torch.mean((q_mse - w) ** 2).item()
    assert mse_mse <= mse_rtn + 1e-12, (mse_mse, mse_rtn)
    assert "best_mse" in meta
    print(f"PASS test_mse_improves_or_equals_rtn_mse rtn={mse_rtn:.6e} mse={mse_mse:.6e}")


def test_rotation_orthogonal():
    R = make_orthogonal(64, seed=0)
    I = R.T @ R
    err = torch.max(torch.abs(I - torch.eye(64))).item()
    assert err < 1e-5, err
    print(f"PASS test_rotation_orthogonal err={err:.2e}")


def test_rotate_unrotate_identity_without_quant():
    w = torch.randn(32, 64)
    R = make_orthogonal(64, seed=1)
    wr = rotate_weight(w, R)
    back = unrotate_weight(wr, R)
    err = torch.max(torch.abs(back - w)).item()
    assert err < 1e-5, err
    print(f"PASS test_rotate_unrotate_identity_without_quant err={err:.2e}")


def test_seeds_differ():
    R0 = make_orthogonal(32, seed=0)
    R1 = make_orthogonal(32, seed=1)
    assert not torch.allclose(R0, R1)
    print("PASS test_seeds_differ")


if __name__ == "__main__":
    test_rtn_shapes()
    test_mse_improves_or_equals_rtn_mse()
    test_rotation_orthogonal()
    test_rotate_unrotate_identity_without_quant()
    test_seeds_differ()
    print("ALL UNIT TESTS PASSED")
