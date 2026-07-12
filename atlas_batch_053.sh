#!/bin/bash
set -e
cd /workspace
run() { python3 atlas_run.py "$@"; }

echo "=== exp-053 tri_scale @7200 (ataque ao recorde 92.90%) ==="
run --id exp-053 --name tri_scale_ls_wc_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | tri-scale escala 1.5x." --lesson "Regra 3 do tri_scale." --seeds 3 -- \
  --steps 7200 --eval_every 2400 --mixing tri_scale_blend --label_smoothing 0.1 \
  --warmup_steps 150 --scheduler cosine

echo "=== exp-054 ViT token_blend compute-justo (3400 steps = mesmo wall-time que attn 2400) ==="
run --id exp-054 --name vit_token_blend_3400_computefair --verdict INCONCLUSIVA \
  --novelty "NOVEL | comparação compute-justa: token_blend 50sps vs attn 35.5sps => 3400 steps no mesmo tempo." \
  --lesson "Regra 2: mesmo budget de COMPUTE (wall-time)." --seeds 3 -- \
  --steps 3400 --eval_every 1133 --arch vit --vit_mixer token_blend \
  --label_smoothing 0.1 --warmup_steps 140 --scheduler cosine

echo "=== BATCH 6 DONE ==="
