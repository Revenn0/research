#!/bin/bash
set -e
cd /workspace
run() { python3 atlas_run.py "$@"; }
LS="--label_smoothing 0.1"
WC="--warmup_steps 100 --scheduler cosine"

echo "=== exp-048 ViT attn baseline @2400 ==="
run --id exp-048 --name vit_attn_2400 --verdict BASELINE \
  --novelty "CONHECIDA | TinyViT self-attention padrão (baseline transformer)." \
  --lesson "Baseline transformer track." --seeds 3 -- \
  --steps 2400 --eval_every 800 --arch vit $LS $WC

echo "=== exp-049 ViT token_blend @2400 ==="
run --id exp-049 --name vit_token_blend_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | token mixing O(N) sem QKV: conv1d depthwise + gate global. vs MLP-Mixer: sem MLP de tokens; vs ConvMixer: gate adaptativo + init identidade." \
  --lesson "local_blend aplicado a tokens." --seeds 3 -- \
  --steps 2400 --eval_every 800 --arch vit --vit_mixer token_blend $LS $WC

echo "=== exp-050 ViT token_blend_ms @2400 ==="
run --id exp-050 --name vit_token_blend_ms_2400 --verdict INCONCLUSIVA \
  --novelty "NOVEL | multi-scale token blend (k3+k5) sobre sequência." \
  --lesson "MultiScale nos tokens." --seeds 3 -- \
  --steps 2400 --eval_every 800 --arch vit --vit_mixer token_blend_ms $LS $WC

echo "=== exp-051 tri_scale+LS+wc @4800 (ataque ao recorde) ==="
run --id exp-051 --name tri_scale_ls_wc_4800 --verdict INCONCLUSIVA \
  --novelty "NOVEL | tri-scale 3+5+7 blend." \
  --lesson "Extensão do campeão." --seeds 3 -- \
  --steps 4800 --eval_every 1600 --mixing tri_scale_blend $LS $WC

echo "=== exp-052 multi_scale+LS+wc+mixup @4800 ==="
run --id exp-052 --name ms_ls_wc_mixup_4800 --verdict INCONCLUSIVA \
  --novelty "RECOMB | campeão + mixup 0.2." \
  --lesson "Stack + augmentation." --seeds 3 -- \
  --steps 4800 --eval_every 1600 --mixing multi_scale_blend $LS $WC --mixup_alpha 0.2

echo "=== BATCH 5 DONE ==="
