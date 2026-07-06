#!/bin/bash
# Retoma batch 100M pulando experimentos já gravados em experiments_log.jsonl
set -e
cd "$(dirname "$0")"

done_id() {
  python3 -c "
import json, sys
done = {json.loads(l)['id'] for l in open('experiments_log.jsonl') if l.strip() and 'exp-100m' in l}
print('yes' if sys.argv[1] in done else 'no')
" "$1"
}

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

maybe_run() {
  local id="$1"
  if [ "$(done_id "$id")" = "yes" ]; then
    echo "=== SKIP $id (já concluído) ==="
    return 0
  fi
  shift
  run "$@"
}

echo "========== RETOMADA BATCH 100M =========="

maybe_run exp-100m-001 --id exp-100m-001 --name p007_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-007 @100M." \
  --lesson "Promissora @100M: warmup+cosine." --seeds 3 -- \
  $SCALE $S1000 $WC

echo "=== exp-100m-002 | exp-011 warmup+cosine ==="
maybe_run exp-100m-002 --id exp-100m-002 --name p011_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-011 @100M." \
  --lesson "Promissora @100M: warmup+cosine." --seeds 3 -- \
  $SCALE $S1000 $WC

echo "=== exp-100m-003 | exp-016 local_blend ==="
maybe_run exp-100m-003 --id exp-100m-003 --name p016_blend_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-016 @100M." \
  --lesson "Promissora @100M: local_blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend

echo "=== exp-100m-004 | exp-018 local_blend ==="
maybe_run exp-100m-004 --id exp-100m-004 --name p018_blend_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-018 @100M." \
  --lesson "Promissora @100M: local_blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend

echo "=== exp-100m-005 | exp-021 local_blend_k5 ==="
maybe_run exp-100m-005 --id exp-100m-005 --name p021_k5_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-021 @100M." \
  --lesson "Promissora @100M: k5 blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5

echo "=== exp-100m-006 | exp-022 laplacian_blend ==="
maybe_run exp-100m-006 --id exp-100m-006 --name p022_laplacian_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-022 @100M." \
  --lesson "Promissora @100M: laplacian." --seeds 3 -- \
  $SCALE $S1000 --mixing laplacian_blend

echo "=== exp-100m-007 | exp-024 blend+warmup+cosine ==="
maybe_run exp-100m-007 --id exp-100m-007 --name p024_blend_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-024 @100M." \
  --lesson "Promissora @100M: blend+wc." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $WC

echo "=== exp-100m-008 | exp-025 blend+LS ==="
maybe_run exp-100m-008 --id exp-100m-008 --name p025_blend_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-025 @100M." \
  --lesson "Promissora @100M: blend+LS." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $LS

echo "=== exp-100m-009 | exp-026 local_blend_k5 ==="
maybe_run exp-100m-009 --id exp-100m-009 --name p026_k5_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-026 @100M." \
  --lesson "Promissora @100M: k5." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5

echo "=== exp-100m-010 | exp-027 blend+LS ex-campeão ==="
maybe_run exp-100m-010 --id exp-100m-010 --name p027_blend_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | campeão exp-027 @100M." \
  --lesson "Ex-campeão promissora @100M." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $LS

echo "=== exp-100m-011 | exp-028 laplacian_blend ==="
maybe_run exp-100m-011 --id exp-100m-011 --name p028_laplacian_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-028 @100M." \
  --lesson "Promissora @100M: laplacian." --seeds 3 -- \
  $SCALE $S1000 --mixing laplacian_blend

echo "=== exp-100m-012 | exp-032 k5+LS ==="
maybe_run exp-100m-012 --id exp-100m-012 --name p032_k5_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-032 @100M." \
  --lesson "Promissora @100M: k5+LS." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5 $LS

echo "=== exp-100m-baseline | referência @100M ==="
maybe_run exp-100m-baseline --id exp-100m-baseline --name baseline_100m --verdict INCONCLUSIVA \
  --novelty "BASELINE | referência @100M para ranking promissoras." \
  --lesson "Baseline escala 100M." --seeds 3 -- \
  $SCALE $S1000

python3 atlas_100m_dashboard.py
echo "========== BATCH 100M RETOMADO / CONCLUÍDO =========="
