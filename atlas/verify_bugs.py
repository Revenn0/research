#!/usr/bin/env python3
"""Active checks for the three known ATL-0032 bugs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from atlas.datasets import consecutive_batch, load_train_val, random_batch
from atlas.trainer import LMTrainConfig, train_lm


def check_vocab() -> bool:
    train_ds, val_ds = load_train_val("atlas/data/tinyshakespeare.txt")
    assert train_ds.stoi == val_ds.stoi
    assert train_ds.vocab_size == val_ds.vocab_size
    print(f"OK vocab: size={train_ds.vocab_size}, shared stoi")
    return True


def check_batch_diversity() -> bool:
    train_ds, _ = load_train_val("atlas/data/tinyshakespeare.txt")
    x, _ = random_batch(train_ds.data, block_size=64, batch_size=16)
    ratio = len(set(x.flatten().tolist())) / x.numel()
    x_bad, _ = consecutive_batch(train_ds.data, 64, 16, start=0)
    bad_ratio = len(set(x_bad.flatten().tolist())) / x_bad.numel()
    per_seq = sum(len(set(x[i].tolist())) for i in range(x.size(0))) / (x.size(0) * x.size(1))
    print(f"random_batch global unique ratio: {ratio:.1%} (per-seq mean {per_seq:.1%})")
    print(f"consecutive_batch global unique ratio (buggy): {bad_ratio:.1%}")
    assert ratio > bad_ratio * 1.3, "random_batch should beat overlapping consecutive batching"
    assert bad_ratio < 0.04, f"buggy batch should stay ~2% global, got {bad_ratio:.1%}"
    return True


def check_causality() -> bool:
    cfg = LMTrainConfig(
        tag="causal-check",
        seed=42,
        total_steps=50,
        eval_every=50,
        mixing="long-conv-fft",
        optimizer="signlion",
        schedule="cosine-warmup",
    )
    result = train_lm(cfg)
    print(f"val_loss @50 steps (long-conv): {result.final_val_loss:.4f}")
    assert result.final_val_loss > 1.0, "Causal leak suspected: val_loss < 1.0"
    return True


def main() -> None:
    check_vocab()
    check_batch_diversity()
    check_causality()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
