#!/usr/bin/env python3
"""Utilitário ATLAS: roda experimento, agrega seeds, grava experiments_log.jsonl."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev


def run_one(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def aggregate(results: list[dict]) -> dict:
    accs = [r["best_val_acc"] for r in results]
    out = {
        "n_seeds": len(results),
        "best_val_acc_mean": mean(accs),
        "best_val_acc_std": stdev(accs) if len(accs) > 1 else 0.0,
        "wall_time_s_mean": mean(r["wall_time_s"] for r in results),
        "steps_per_sec_mean": mean(r["steps_per_sec"] for r in results),
        "per_seed": results,
    }
    return out


def append_log(entry: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--verdict", default="INCONCLUSIVA")
    p.add_argument("--lesson", default="")
    p.add_argument("--novelty", default="")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--log", default="experiments_log.jsonl")
    p.add_argument("train_args", nargs=argparse.REMAINDER, help="args passed to train_baseline.py after --")
    args = p.parse_args()

    train_args = [a for a in args.train_args if a != "--"]
    base_cmd = [sys.executable, "train_baseline.py"] + train_args

    results = []
    for i in range(args.seeds):
        seed = 1000 + i
        cmd = base_cmd + ["--seed", str(seed)]
        results.append(run_one(cmd))

    agg = aggregate(results)
    entry = {
        "id": args.id,
        "name": args.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repro_command": " ".join(base_cmd + ["--seed", "<SEED>"]),
        "seeds": [1000 + i for i in range(args.seeds)],
        "config": results[0]["config"],
        "result": agg,
        "verdict": args.verdict,
        "lesson": args.lesson,
        "novelty_note": args.novelty,
    }
    append_log(entry, Path(args.log))
    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
