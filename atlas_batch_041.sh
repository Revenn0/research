#!/bin/bash
set -e
cd /workspace
run() { python3 atlas_run.py "$@"; }

echo "=== exp-041 k5+LS @4800 (escala 2x do campeão) ==="
run --id exp-041 --name k5_ls_4800 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 2x exp-033." --lesson "Regra 3: budget 4800." --seeds 3 -- \
  --steps 4800 --eval_every 1600 --mixing local_blend_k5 --label_smoothing 0.1

echo "=== exp-042 multi_scale+LS+wc @4800 ==="
run --id exp-042 --name multi_scale_ls_wc_4800 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 2x exp-040." --lesson "Budget 4800 stack." --seeds 3 -- \
  --steps 4800 --eval_every 1600 --mixing multi_scale_blend --label_smoothing 0.1 \
  --warmup_steps 100 --scheduler cosine

echo "=== exp-043 k5+LS hidden256 @2400 ==="
run --id exp-043 --name k5_ls_h256_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | mais capacidade FC + k5+LS." --lesson "hidden_dim=256." --seeds 3 -- \
  --steps 2400 --eval_every 800 --mixing local_blend_k5 --label_smoothing 0.1 --hidden_dim 256

echo "=== exp-044 cascade_blend+LS @2400 ==="
run --id exp-044 --name cascade_blend_ls_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | blend duplo em série." --lesson "cascade_blend mixing." --seeds 3 -- \
  --steps 2400 --eval_every 800 --mixing cascade_blend --label_smoothing 0.1

echo "=== exp-045 k5+LS dropout005 @2400 ==="
run --id exp-045 --name k5_ls_do005_2400 --verdict INCONCLUSIVA \
  --novelty "RECOMB | menos dropout + k5+LS." --lesson "dropout=0.05." --seeds 3 -- \
  --steps 2400 --eval_every 800 --mixing local_blend_k5 --label_smoothing 0.1 --dropout 0.05

echo "=== BATCH 4 DONE ==="
