#!/bin/bash
# ============================================================================
# ATLAS 100M — 12 PROMISSORAS + baseline em ~108M parâmetros (CPU)
#
# Escala: width_mult=16, hidden_dim=2048 → ~107.5M params (SmallCNN FashionMNIST)
# Budget: 1000 steps × 3 seeds (comparável ao orçamento promissora original)
# ETA CPU (~0.4 steps/s): ~35–40 min/seed → ~22–26 h total
#
# Uso:  ./atlas_batch_100m.sh 2>&1 | tee batch_100m.log
# ============================================================================
set -e
cd "$(dirname "$0")"

echo "========== VERIFICAÇÃO ESCALA 100M =========="
python3 - << 'EOF'
from train_baseline import SmallCNN, TrainConfig
cfg = TrainConfig(width_mult=16.0, hidden_dim=2048)
n = sum(p.numel() for p in SmallCNN(cfg).parameters())
print(f"Parâmetros: {n/1e6:.2f}M")
assert 95e6 <= n <= 115e6, f"Fora da faixa 100M: {n/1e6:.1f}M"
print("OK: modelo dentro da faixa ~100M")
EOF

run() {
  python3 atlas_run.py \
    --result_dir results/100m \
    --refresh_dashboard \
    "$@"
  python3 atlas_100m_dashboard.py
}

SCALE="--width_mult 16.0 --hidden_dim 2048 --batch_size 64 --lr 0.001 --weight_decay 1e-4 --device cpu --num_workers 2"
S1000="--steps 1000 --eval_every 250 --log_every 200"
WC="--warmup_steps 100 --scheduler cosine"
LS="--label_smoothing 0.1"

echo "========== FASE 1: 12 PROMISSORAS @100M, 1000 steps =========="

echo "=== exp-100m-001 | exp-007 warmup+cosine ==="
run --id exp-100m-001 --name p007_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-007 @100M." \
  --lesson "Promissora @100M: warmup+cosine." --seeds 3 -- \
  $SCALE $S1000 $WC

echo "=== exp-100m-002 | exp-011 warmup+cosine ==="
run --id exp-100m-002 --name p011_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-011 @100M." \
  --lesson "Promissora @100M: warmup+cosine." --seeds 3 -- \
  $SCALE $S1000 $WC

echo "=== exp-100m-003 | exp-016 local_blend ==="
run --id exp-100m-003 --name p016_blend_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-016 @100M." \
  --lesson "Promissora @100M: local_blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend

echo "=== exp-100m-004 | exp-018 local_blend ==="
run --id exp-100m-004 --name p018_blend_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-018 @100M." \
  --lesson "Promissora @100M: local_blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend

echo "=== exp-100m-005 | exp-021 local_blend_k5 ==="
run --id exp-100m-005 --name p021_k5_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-021 @100M." \
  --lesson "Promissora @100M: k5 blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5

echo "=== exp-100m-006 | exp-022 laplacian_blend ==="
run --id exp-100m-006 --name p022_laplacian_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-022 @100M." \
  --lesson "Promissora @100M: laplacian." --seeds 3 -- \
  $SCALE $S1000 --mixing laplacian_blend

echo "=== exp-100m-007 | exp-024 blend+warmup+cosine ==="
run --id exp-100m-007 --name p024_blend_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-024 @100M." \
  --lesson "Promissora @100M: blend+wc." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $WC

echo "=== exp-100m-008 | exp-025 blend+LS ==="
run --id exp-100m-008 --name p025_blend_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-025 @100M." \
  --lesson "Promissora @100M: blend+LS." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $LS

echo "=== exp-100m-009 | exp-026 local_blend_k5 ==="
run --id exp-100m-009 --name p026_k5_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-026 @100M." \
  --lesson "Promissora @100M: k5." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5

echo "=== exp-100m-010 | exp-027 blend+LS ex-campeão ==="
run --id exp-100m-010 --name p027_blend_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | campeão exp-027 @100M." \
  --lesson "Ex-campeão promissora @100M." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $LS

echo "=== exp-100m-011 | exp-028 laplacian_blend ==="
run --id exp-100m-011 --name p028_laplacian_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-028 @100M." \
  --lesson "Promissora @100M: laplacian." --seeds 3 -- \
  $SCALE $S1000 --mixing laplacian_blend

echo "=== exp-100m-012 | exp-032 k5+LS ==="
run --id exp-100m-012 --name p032_k5_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-032 @100M." \
  --lesson "Promissora @100M: k5+LS." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5 $LS

echo "=== exp-100m-baseline | referência @100M ==="
run --id exp-100m-baseline --name baseline_100m --verdict INCONCLUSIVA \
  --novelty "BASELINE | referência @100M para ranking promissoras." \
  --lesson "Baseline escala 100M." --seeds 3 -- \
  $SCALE $S1000

echo "========== RANKING 100M =========="
python3 - << 'EOF'
import json
rows = []
baseline = None
for line in open("experiments_log.jsonl"):
    e = json.loads(line)
    if not e["id"].startswith("exp-100m"):
        continue
    r = e["result"]
    acc = r["best_val_acc_mean"]
    if e["id"] == "exp-100m-baseline":
        baseline = acc
    rows.append((acc, r["best_val_acc_std"], e["id"], e["name"], e.get("novelty_note", "")))
rows.sort(reverse=True)
print(f"Baseline @100M: {baseline*100:.2f}%" if baseline else "Baseline: N/A")
print()
for acc, std, eid, name, nov in rows:
    delta = (acc - baseline) * 100 if baseline else 0
    print(f"{acc*100:6.2f}% ± {std*100:.2f}%  ({delta:+.2f}pp)  {eid:20} {name}")
EOF

echo "========== BATCH 100M DONE =========="
