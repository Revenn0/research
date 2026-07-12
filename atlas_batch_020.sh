#!/bin/bash
# ATLAS batch exp-020..025
set -e
cd /workspace

run() {
  python3 atlas_run.py "$@"
}

echo "=== exp-020 local_blend_var ==="
run --id exp-020 --name local_blend_var --verdict INCONCLUSIVA \
  --novelty "NOVEL | Gate por variância espacial em vez de média." \
  --lesson "Mutação local_blend: gate=sigmoid(4*var)." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing local_blend_var

echo "=== exp-021 local_blend_k5 ==="
run --id exp-021 --name local_blend_k5 --verdict INCONCLUSIVA \
  --novelty "NOVEL | Local blend com kernel depthwise 5x5." \
  --lesson "Mutação local_blend: receptive field maior." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing local_blend_k5

echo "=== exp-022 laplacian_blend ==="
run --id exp-022 --name laplacian_blend --verdict INCONCLUSIVA \
  --novelty "NOVEL | Depthwise Laplaciano fixo + gate média." \
  --lesson "Mutação: filtro de borda em vez de dirac." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing laplacian_blend

echo "=== exp-023 dual_gate_blend ==="
run --id exp-023 --name dual_gate_blend --verdict INCONCLUSIVA \
  --novelty "NOVEL | Gate = sigmoid(mean)*sigmoid(4*var)." \
  --lesson "Mutação: gate dual mean+variance." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing dual_gate_blend

echo "=== exp-024 local_blend_warmup_cosine ==="
run --id exp-024 --name local_blend_warmup_cosine --verdict INCONCLUSIVA \
  --novelty "RECOMB | local_blend NOVEL + warmup+cosine RECOMB." \
  --lesson "Combo dos dois campeões." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing local_blend --warmup_steps 100 --scheduler cosine

echo "=== exp-025 local_blend_ls ==="
run --id exp-025 --name local_blend_ls --verdict INCONCLUSIVA \
  --novelty "RECOMB | local_blend + label_smoothing 0.1." \
  --lesson "Combo blend + regularização." --seeds 3 -- \
  --steps 1200 --eval_every 400 --mixing local_blend --label_smoothing 0.1

echo "=== BATCH DONE ==="
