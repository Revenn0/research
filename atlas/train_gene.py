#!/usr/bin/env python3
"""Configurable gene trainer for ATLAS char-level LM experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.trainer import LMTrainConfig, append_experiment_log, result_to_log_entry, train_lm


def parse_args() -> LMTrainConfig:
    p = argparse.ArgumentParser(description="ATLAS gene char LM")
    p.add_argument("--tag", default="gene")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_steps", type=int, default=2760)
    p.add_argument("--eval_every", type=int, default=460)
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mixing", default="long-conv-fft")
    p.add_argument("--norm", default="layernorm")
    p.add_argument("--activation", default="gelu")
    p.add_argument("--optimizer", default="signlion")
    p.add_argument("--schedule", default="cosine-warmup")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--data_path", default="atlas/data/tinyshakespeare.txt")
    args = p.parse_args()
    return LMTrainConfig(
        tag=args.tag,
        seed=args.seed,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        batch=args.batch,
        dropout=args.dropout,
        mixing=args.mixing,
        norm=args.norm,
        activation=args.activation,
        optimizer=args.optimizer,
        schedule=args.schedule,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        data_path=args.data_path,
    )


def main() -> None:
    cfg = parse_args()
    cmd = " ".join(sys.argv)
    result = train_lm(cfg)
    append_experiment_log(result_to_log_entry(result, cmd))
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
