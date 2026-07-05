#!/bin/bash
set -e
cd /workspace
run() { python3 atlas_run.py "$@"; }

# Scale tests (regra 3)
run --id exp-026 --name local_blend_k5_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala exp-021 k5." --lesson "Scale test k5 blend." --seeds 3 -- \
  --steps 2400 --eval_every 800 --mixing local_blend_k5

run --id exp-027 --name local_blend_ls_2400 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala exp-025." --lesson "Scale test blend+LS." --seeds 3 -- \
  --steps 2400 --eval_every 800 --mixing local_blend --label_smoothing 0.1

run --id exp-028 --name laplacian_blend_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala exp-022." --lesson "Scale test laplacian." --seeds 3 -- \
  --steps 2400 --eval_every 800 --mixing laplacian_blend

# Novos NOVEL
run --id exp-029 --name rms_free_norm --verdict INCONCLUSIVA \
  --novelty "NOVEL | RMS sem afim por canal. vs LN: sem shift/scale." --lesson "Norm param-free." --seeds 3 -- \
  --steps 1200 --eval_every 400 --norm rms_free

run --id exp-030 --name grad_shrink --verdict INCONCLUSIVA \
  --novelty "NOVEL | g*=sigmoid(|g|) antes Adam. vs GC: não centraliza." --lesson "Grad shrink optimizer." --seeds 3 -- \
  --steps 1200 --eval_every 400 --optimizer grad_shrink

run --id exp-031 --name blend_k5_warmup_cosine --verdict INCONCLUSIVA \
  --novelty "RECOMB | k5 blend + warmup+cosine." --lesson "Combo top mutações." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing local_blend_k5 --warmup_steps 100 --scheduler cosine

run --id exp-032 --name blend_k5_ls --verdict INCONCLUSIVA \
  --novelty "RECOMB | k5 blend + label smoothing." --lesson "Combo k5+LS." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing local_blend_k5 --label_smoothing 0.1

echo "=== BATCH 2 DONE ==="
