#!/bin/bash
# Batch escala máxima: 12 PROMISSORAS @7200 steps (3 seeds) + baseline + top-4 @9600
set -e
cd /workspace
run() { python3 atlas_run.py "$@"; }

S7200="--steps 7200 --eval_every 2400"
S9600="--steps 9600 --eval_every 3200"
WC="--warmup_steps 150 --scheduler cosine"
LS="--label_smoothing 0.1"

echo "========== FASE 1: 12 PROMISSORAS @7200 (Regra 3 — escala 3–6×) =========="

echo "=== exp-059 | exp-007 warmup+cosine @7200 ==="
run --id exp-059 --name p007_warmup_cosine_7200 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 6× do exp-007 PROMISSORA." \
  --lesson "Promissora @7200: warmup+cosine." --seeds 3 -- \
  $S7200 --warmup_steps 150 --scheduler cosine

echo "=== exp-060 | exp-011 warmup+cosine @7200 ==="
run --id exp-060 --name p011_warmup_cosine_7200 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 3× do exp-011 PROMISSORA." \
  --lesson "Promissora @7200: warmup+cosine." --seeds 3 -- \
  $S7200 --warmup_steps 150 --scheduler cosine

echo "=== exp-061 | exp-016 local_blend @7200 ==="
run --id exp-061 --name p016_local_blend_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 6× do exp-016 PROMISSORA." \
  --lesson "Promissora @7200: local_blend." --seeds 3 -- \
  $S7200 --mixing local_blend

echo "=== exp-062 | exp-018 local_blend @7200 ==="
run --id exp-062 --name p018_local_blend_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 3× do exp-018 PROMISSORA." \
  --lesson "Promissora @7200: local_blend." --seeds 3 -- \
  $S7200 --mixing local_blend

echo "=== exp-063 | exp-021 local_blend_k5 @7200 ==="
run --id exp-063 --name p021_k5_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 6× do exp-021 PROMISSORA." \
  --lesson "Promissora @7200: k5 blend." --seeds 3 -- \
  $S7200 --mixing local_blend_k5

echo "=== exp-064 | exp-022 laplacian_blend @7200 ==="
run --id exp-064 --name p022_laplacian_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 6× do exp-022 PROMISSORA." \
  --lesson "Promissora @7200: laplacian." --seeds 3 -- \
  $S7200 --mixing laplacian_blend

echo "=== exp-065 | exp-024 blend+warmup+cosine @7200 ==="
run --id exp-065 --name p024_blend_wc_7200 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 6× do exp-024 PROMISSORA." \
  --lesson "Promissora @7200: blend+wc." --seeds 3 -- \
  $S7200 --mixing local_blend --warmup_steps 150 --scheduler cosine

echo "=== exp-066 | exp-025 blend+LS @7200 ==="
run --id exp-066 --name p025_blend_ls_7200 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 6× do exp-025 PROMISSORA." \
  --lesson "Promissora @7200: blend+LS." --seeds 3 -- \
  $S7200 --mixing local_blend --label_smoothing 0.1

echo "=== exp-067 | exp-026 k5 @7200 ==="
run --id exp-067 --name p026_k5_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 3× do exp-026 PROMISSORA." \
  --lesson "Promissora @7200: k5 blend." --seeds 3 -- \
  $S7200 --mixing local_blend_k5

echo "=== exp-068 | exp-027 blend+LS @7200 (ex-campeão) ==="
run --id exp-068 --name p027_blend_ls_7200 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 3× do exp-027 campeão PROMISSORA (91.78%)." \
  --lesson "Ex-campeão @7200." --seeds 3 -- \
  $S7200 --mixing local_blend --label_smoothing 0.1

echo "=== exp-069 | exp-028 laplacian @7200 ==="
run --id exp-069 --name p028_laplacian_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 3× do exp-028 PROMISSORA." \
  --lesson "Promissora @7200: laplacian." --seeds 3 -- \
  $S7200 --mixing laplacian_blend

echo "=== exp-070 | exp-032 k5+LS @7200 ==="
run --id exp-070 --name p032_k5_ls_7200 --verdict INCONCLUSIVA \
  --novelty "RECOMB | escala 6× do exp-032 PROMISSORA." \
  --lesson "Promissora @7200: k5+LS." --seeds 3 -- \
  $S7200 --mixing local_blend_k5 --label_smoothing 0.1

echo "=== exp-071 | baseline puro @7200 (referência escala) ==="
run --id exp-071 --name baseline_7200 --verdict INCONCLUSIVA \
  --novelty "BASELINE | referência @7200 para ranking promissoras." \
  --lesson "Baseline escala máxima fase 1." --seeds 3 -- \
  $S7200

echo "========== FASE 2: TOP-4 PROMISSORAS @9600 (escala absoluta) =========="

echo "=== exp-072 | exp-027 blend+LS @9600 ==="
run --id exp-072 --name p027_blend_ls_9600 --verdict INCONCLUSIVA \
  --novelty "RECOMB | push escala exp-027." \
  --lesson "Top promissora @9600." --seeds 3 -- \
  $S9600 --mixing local_blend --label_smoothing 0.1

echo "=== exp-073 | exp-026 k5 @9600 ==="
run --id exp-073 --name p026_k5_9600 --verdict INCONCLUSIVA \
  --novelty "NOVEL | push escala exp-026." \
  --lesson "Top promissora k5 @9600." --seeds 3 -- \
  $S9600 --mixing local_blend_k5

echo "=== exp-074 | exp-018 local_blend @9600 ==="
run --id exp-074 --name p018_blend_9600 --verdict INCONCLUSIVA \
  --novelty "NOVEL | push escala exp-018." \
  --lesson "Top promissora blend @9600." --seeds 3 -- \
  $S9600 --mixing local_blend

echo "=== exp-075 | exp-070 k5+LS @9600 (melhor stack RECOMB promissora) ==="
run --id exp-075 --name p032_k5_ls_9600 --verdict INCONCLUSIVA \
  --novelty "RECOMB | k5+LS push @9600." \
  --lesson "Stack RECOMB promissora escala max." --seeds 3 -- \
  $S9600 --mixing local_blend_k5 --label_smoothing 0.1

echo "=== exp-076 | baseline @9600 ==="
run --id exp-076 --name baseline_9600 --verdict INCONCLUSIVA \
  --novelty "BASELINE | referência @9600." \
  --lesson "Baseline escala máxima fase 2." --seeds 3 -- \
  $S9600

echo "========== BATCH PROMISSORAS ESCALA MÁXIMA DONE =========="
