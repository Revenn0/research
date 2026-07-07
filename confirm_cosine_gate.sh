#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
SCALE="--width_mult 16.0 --hidden_dim 2048 --batch_size 64 --lr 0.001 --weight_decay 1e-4 --device cpu --num_workers 2"
STEPS="--steps 1000 --eval_every 250 --log_every 200"
WC="--warmup_steps 100 --scheduler cosine"
python3 atlas_run.py \
  --id explore-cosine-gate-3s \
  --name cosine_gate_blend_confirm \
  --seeds 3 \
  --log explore_results.jsonl \
  --result_dir results/explore \
  --no_refresh_dashboard \
  --lesson "confirm cosine_gate @100M 3 seeds" \
  -- $SCALE $STEPS --mixing cosine_gate_blend $WC

python3 << 'PY'
import json
from pathlib import Path
from statistics import mean, stdev
entries = [json.loads(l) for l in Path("explore_results.jsonl").read_text().splitlines() if l.strip()]
base = next(e for e in entries if e["id"]=="explore-baseline")
cg = next(e for e in entries if e["id"]=="explore-cosine-gate-3s")
b = base["result"]["best_val_acc_mean"]
c = cg["result"]["best_val_acc_mean"]
s = cg["result"]["best_val_acc_std"]
d = (c-b)*100
print(f"CONFIRM: cosine_gate {c*100:.2f}% ± {s*100:.2f}% | Δ baseline {d:+.2f}pp")
with open("explore_live.txt","a") as f:
    f.write(f"\nCONFIRM 3s: {c*100:.2f}% ± {s*100:.2f}% | Δ {d:+.2f}pp\n")
PY
