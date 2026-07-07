#!/bin/bash
# Exploração autônoma ATLAS visão @100M — VM 15GB CPU
set -euo pipefail
cd "$(dirname "$0")"

SCALE="--width_mult 16.0 --hidden_dim 2048 --batch_size 64 --lr 0.001 --weight_decay 1e-4 --device cpu --num_workers 2"
STEPS="--steps 1000 --eval_every 250 --log_every 200"
WC="--warmup_steps 100 --scheduler cosine"
LS="--label_smoothing 0.1"
RESULT_DIR="results/explore"
LOG="explore_results.jsonl"
mkdir -p "$RESULT_DIR"

run_exp() {
  local id="$1" name="$2" extra="$3"
  echo ""
  echo "========== $id | $name =========="
  python3 atlas_run.py \
    --id "$id" --name "$name" \
    --seeds 1 --log "$LOG" \
    --result_dir "$RESULT_DIR" \
    --no_refresh_dashboard \
    --lesson "explore @100M seed42" \
    -- \
    $SCALE $STEPS $extra --seed 42
}

echo "=== ATLAS vision explore — $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee explore_live.txt

run_exp "explore-baseline" "baseline_plain" ""
run_exp "explore-wc" "warmup_cosine" "$WC"
run_exp "explore-ms-ls-wc" "multi_scale_ls_wc" "--mixing multi_scale_blend $LS $WC"
run_exp "explore-tri-ls-wc" "tri_scale_ls_wc" "--mixing tri_scale_blend $LS $WC"
run_exp "explore-local-blend" "local_blend" "--mixing local_blend"
run_exp "explore-cosine-gate" "cosine_gate_blend" "--mixing cosine_gate_blend $WC"
run_exp "explore-hybrid" "hybrid_blend_arch" "--arch hybrid_blend $LS $WC"
run_exp "explore-mixup-ms" "mixup_multi_scale" "--mixing multi_scale_blend --mixup_alpha 0.2 $WC"

echo "" | tee -a explore_live.txt
echo "=== RANKING ===" | tee -a explore_live.txt
python3 << 'PY' | tee -a explore_live.txt
import json
from pathlib import Path

log = Path("explore_results.jsonl")
entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
base = next((e for e in entries if e["id"] == "explore-baseline"), None)
base_acc = base["result"]["best_val_acc_mean"] if base else None

rows = []
for e in entries:
    acc = e["result"]["best_val_acc_mean"]
    delta = (acc - base_acc) * 100 if base_acc else 0
    rows.append((acc, delta, e["id"], e.get("lesson", "")))
rows.sort(reverse=True)

print(f"{'Rank':<5} {'ID':<22} {'Acc':>8} {'Δ base':>8}")
print("-" * 50)
for i, (acc, delta, eid, _) in enumerate(rows, 1):
    print(f"{i:<5} {eid:<22} {acc*100:7.2f}% {delta:+7.2f}pp")

best = rows[0]
if base_acc and best[1] >= 0.30:
    print(f"\n>>> INTERESSANTE: {best[2]} ganha {best[1]:+.2f}pp vs baseline")
elif base_acc and best[1] >= 0.10:
    print(f"\n>>> SINAL FRACO: {best[2]} +{best[1]:.2f}pp — precisa mais seeds")
else:
    print(f"\n>>> Nada forte vs baseline neste screening")
PY

echo "=== FIM $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a explore_live.txt
