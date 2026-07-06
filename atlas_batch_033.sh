#!/bin/bash
# ATLAS batch exp-033+ — meta: superar top-5 (>91.78% @2400)
set -e
cd /workspace
S="--steps 2400 --eval_every 800"
LS="--label_smoothing 0.1"
WC="--warmup_steps 100 --scheduler cosine"
run() { python3 atlas_run.py "$@"; }

echo "=== exp-033 k5+LS @2400 (candidato óbvio) ==="
run --id exp-033 --name k5_ls_2400 --verdict INCONCLUSIVA \
  --novelty "RECOMB | k5 NOVEL + LS, nunca testado @2400." \
  --lesson "Combo mais forte em teoria." --seeds 3 -- \
  $S --mixing local_blend_k5 $LS

echo "=== exp-034 multi_scale+LS @2400 ==="
run --id exp-034 --name multi_scale_ls_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | fusão paralela 3x3+5x5 + LS." \
  --lesson "Multi-scale blend revolucionário." --seeds 3 -- \
  $S --mixing multi_scale_blend $LS

echo "=== exp-035 multi_scale @2400 ==="
run --id exp-035 --name multi_scale_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | multi_scale_blend isolado." \
  --lesson "Baseline multi-escala." --seeds 3 -- \
  $S --mixing multi_scale_blend

echo "=== exp-036 cosine_gate+LS @2400 ==="
run --id exp-036 --name cosine_gate_ls_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | gate cossim(x,conv(x)) + LS." \
  --lesson "Gate por similaridade direcional." --seeds 3 -- \
  $S --mixing cosine_gate_blend $LS

echo "=== exp-037 entropy_gate+LS @2400 ==="
run --id exp-037 --name entropy_gate_ls_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | gate entropia espacial + LS." \
  --lesson "Gate informacional." --seeds 3 -- \
  $S --mixing entropy_gate_blend $LS

echo "=== exp-038 post_act_blend+LS @2400 ==="
run --id exp-038 --name post_act_ls_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | blend pós-ReLU + LS." \
  --lesson "Mixing no espaço de ativação." --seeds 3 -- \
  $S --mixing post_act_blend $LS

echo "=== exp-039 k5+LS+wc @2400 ==="
run --id exp-039 --name k5_ls_wc_2400 --verdict INCONCLUSIVA \
  --novelty "RECOMB | triple stack campeões." \
  --lesson "k5+LS+warmup+cosine." --seeds 3 -- \
  $S --mixing local_blend_k5 $LS $WC

echo "=== exp-040 multi_scale+LS+wc @2400 ==="
run --id exp-040 --name multi_scale_ls_wc_2400 --verdict INCONCLUSIVA \
  --novelty "RECOMB | multi_scale+LS+cosine." \
  --lesson "Stack agressivo." --seeds 3 -- \
  $S --mixing multi_scale_blend $LS $WC

echo "=== BATCH 3 DONE ==="
