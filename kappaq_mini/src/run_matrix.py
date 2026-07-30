#!/usr/bin/env python3
"""Serial runner for C-MINI-LADDER-v1 arm×seed matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["A0", "A1", "A2", "A3"]
SEEDS = [0, 1, 2]


def main() -> int:
    py = sys.executable
    script = ROOT / "src" / "run_arm.py"
    failures = []
    for arm in ARMS:
        for seed in SEEDS:
            cmd = [py, str(script), "--arm", arm, "--seed", str(seed)]
            print("=" * 60, flush=True)
            print("RUN", " ".join(cmd), flush=True)
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                failures.append((arm, seed, rc))
                print(f"FAIL arm={arm} seed={seed} rc={rc}", flush=True)
            else:
                print(f"OK arm={arm} seed={seed}", flush=True)
    print("=" * 60, flush=True)
    print(f"DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
