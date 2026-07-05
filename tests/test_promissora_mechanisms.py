#!/usr/bin/env python3
"""Testes unitários dos mecanismos das 12 PROMISSORAS.

Cobertura por promissora:
  exp-007/011  -> warmup+cosine scheduler
  exp-016/018  -> local_blend (LocalBlend k3, gate mean)
  exp-021/026  -> local_blend_k5 (LocalBlend k5)
  exp-022/028  -> laplacian_blend (LocalBlend gate laplacian)
  exp-024      -> local_blend + warmup+cosine
  exp-025/027  -> local_blend + label_smoothing
  exp-032      -> local_blend_k5 + label_smoothing

Roda com pytest OU diretamente: python3 tests/test_promissora_mechanisms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from train_baseline import (
    LocalBlend,
    SmallCNN,
    TrainConfig,
    lr_at_step,
    make_mixing,
)

TOL = 1e-5


# ---------- LocalBlend (exp-016/018/021/026) ----------

def test_local_blend_shape_preserved():
    for kernel in (3, 5):
        m = LocalBlend(16, kernel=kernel, gate_mode="mean")
        x = torch.randn(4, 16, 14, 14)
        assert m(x).shape == x.shape


def test_local_blend_init_behavior():
    """Documenta o comportamento REAL do init.

    `nn.init.dirac_` em peso depthwise (C,1,k,k) só põe identidade no canal 0;
    canais 1..C-1 iniciam ZERADOS. Logo no init:
      - canal 0:   local == x  => out == x (identidade)
      - canais >0: local == 0  => out == (1-gate)*x (atenuação)
    Este é o mecanismo que produziu os resultados campeões — o teste fixa
    esse comportamento para detectar regressões acidentais.
    """
    for kernel in (3, 5):
        m = LocalBlend(16, kernel=kernel, gate_mode="mean")
        x = torch.randn(4, 16, 14, 14)
        out = m(x)
        # canal 0 é identidade exata
        assert torch.allclose(out[:, 0], x[:, 0], atol=TOL), f"k{kernel}: canal 0 deveria ser identidade"
        # canais >0: out = (1-gate)*x
        gate = m._gate(x)
        expected = (1.0 - gate[:, 1:]) * x[:, 1:]
        assert torch.allclose(out[:, 1:], expected, atol=TOL), f"k{kernel}: canais >0 deveriam ser (1-gate)*x"


def test_local_blend_gate_range():
    m = LocalBlend(8, kernel=3, gate_mode="mean")
    x = torch.randn(2, 8, 10, 10) * 5
    gate = m._gate(x)
    assert gate.min() >= 0.0 and gate.max() <= 1.0
    assert gate.shape == (2, 8, 1, 1), "gate deve ser escalar por canal"


def test_local_blend_gradient_flows():
    m = LocalBlend(8, kernel=3, gate_mode="mean")
    x = torch.randn(2, 8, 10, 10, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert m.dw.weight.grad is not None and m.dw.weight.grad.abs().sum() > 0


# ---------- laplacian_blend (exp-022/028) ----------

def test_laplacian_blend_kernel_is_laplacian():
    m = LocalBlend(4, kernel=3, gate_mode="laplacian")
    expected = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    for c in range(4):
        assert torch.allclose(m.dw.weight[c, 0], expected), "kernel deve ser Laplaciano"


def test_laplacian_blend_responds_to_edges():
    """Laplaciano de entrada constante = 0 => saída = (1-gate)*x."""
    m = LocalBlend(2, kernel=3, gate_mode="laplacian")
    x = torch.ones(1, 2, 8, 8)
    out = m(x)
    # interior: local=0 (constante), então out = (1-gate)*x < x
    interior = out[..., 2:-2, 2:-2]
    assert (interior < 1.0).all(), "blend deve atenuar região constante"


# ---------- warmup+cosine (exp-007/011/024) ----------

def test_warmup_ramps_linearly():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, scheduler="cosine")
    lr0 = lr_at_step(cfg, 0, 1000)
    lr50 = lr_at_step(cfg, 50, 1000)
    lr99 = lr_at_step(cfg, 99, 1000)
    assert lr0 < lr50 < lr99 <= cfg.lr
    assert abs(lr50 - cfg.lr * 51 / 100) < TOL


def test_cosine_decays_to_zero():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, scheduler="cosine")
    lr_mid = lr_at_step(cfg, 550, 1000)
    lr_end = lr_at_step(cfg, 1000, 1000)
    assert lr_mid < cfg.lr
    assert lr_end < 1e-6, f"cosine deve terminar ~0, got {lr_end}"


def test_cosine_monotonic_after_warmup():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, scheduler="cosine")
    lrs = [lr_at_step(cfg, s, 1000) for s in range(100, 1001, 50)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "decay deve ser monotônico"


# ---------- label_smoothing (exp-025/027/032) ----------

def test_label_smoothing_penalizes_overconfidence():
    logits = torch.zeros(1, 10)
    logits[0, 3] = 100.0  # confiança extrema
    y = torch.tensor([3])
    ce = F.cross_entropy(logits, y, label_smoothing=0.0)
    ce_ls = F.cross_entropy(logits, y, label_smoothing=0.1)
    assert ce < 1e-4
    assert ce_ls > ce, "LS deve penalizar overconfidence"


def test_label_smoothing_keeps_argmax_learnable():
    torch.manual_seed(0)
    logits = torch.randn(32, 10, requires_grad=True)
    y = torch.randint(0, 10, (32,))
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ---------- make_mixing factory ----------

def test_make_mixing_promissora_names():
    for name in ("local_blend", "local_blend_k5", "laplacian_blend"):
        m = make_mixing(name, 16)
        x = torch.randn(2, 16, 14, 14)
        assert m(x).shape == x.shape, f"{name} quebrou shape"


def test_make_mixing_none_is_identity():
    m = make_mixing("none", 16)
    x = torch.randn(2, 16, 14, 14)
    assert torch.equal(m(x), x)


# ---------- integração: SmallCNN com cada config promissora ----------

PROMISSORA_CONFIGS = {
    "exp-007/011_warmup_cosine": dict(warmup_steps=100, scheduler="cosine"),
    "exp-016/018_local_blend": dict(mixing="local_blend"),
    "exp-021/026_k5": dict(mixing="local_blend_k5"),
    "exp-022/028_laplacian": dict(mixing="laplacian_blend"),
    "exp-024_blend_wc": dict(mixing="local_blend", warmup_steps=100, scheduler="cosine"),
    "exp-025/027_blend_ls": dict(mixing="local_blend", label_smoothing=0.1),
    "exp-032_k5_ls": dict(mixing="local_blend_k5", label_smoothing=0.1),
}


def test_smallcnn_forward_all_promissoras():
    x = torch.randn(4, 1, 28, 28)
    for name, kw in PROMISSORA_CONFIGS.items():
        cfg = TrainConfig(**kw)
        model = SmallCNN(cfg)
        logits = model(x)
        assert logits.shape == (4, 10), f"{name}: shape errado"
        assert torch.isfinite(logits).all(), f"{name}: logits não-finitos"


def test_smallcnn_overfits_fixed_batch_all_promissoras():
    """20 passos em batch fixo devem reduzir a loss claramente (sanity de treino)."""
    for name, kw in PROMISSORA_CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        cfg = TrainConfig(**kw)
        model = SmallCNN(cfg)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ls = kw.get("label_smoothing", 0.0)
        loss0 = F.cross_entropy(model(x), y, label_smoothing=ls).item()
        for _ in range(20):
            loss = F.cross_entropy(model(x), y, label_smoothing=ls)
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_end = F.cross_entropy(model(x), y, label_smoothing=ls).item()
        assert loss_end < loss0 * 0.7, f"{name}: loss não caiu o suficiente ({loss0:.4f} -> {loss_end:.4f})"


def _run_all():
    fns = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"FAIL {name}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
