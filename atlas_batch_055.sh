#!/bin/bash
set -e
cd /workspace
run() { python3 atlas_run.py "$@"; }
LS="--label_smoothing 0.1"
WC="--warmup_steps 100 --scheduler cosine"

echo "=== exp-055 HYBRID tri_scale+LS+wc @4800 (REVOLUCIONÁRIO) ==="
run --id exp-055 --name hybrid_tri_ls_wc_4800 --verdict INCONCLUSIVA \
  --novelty "NOVEL REVOLUCIONÁRIO | HybridBlendNet: tri_scale spatial + TokenBlendMS sobre 49 tokens espaciais. vs ViT: features CNN; vs CNN: mixing O(N) entre posições; vs ConvMixer: gate adaptativo+multi-scale." \
  --lesson "Arquitetura híbrida ATLAS." --seeds 3 -- \
  --steps 4800 --eval_every 1600 --arch hybrid_blend --mixing tri_scale_blend $LS $WC

echo "=== exp-056 HYBRID multi_scale+LS+wc @4800 ==="
run --id exp-056 --name hybrid_ms_ls_wc_4800 --verdict INCONCLUSIVA \
  --novelty "NOVEL | Hybrid com multi_scale (campeão CNN)." \
  --lesson "Hybrid + multi_scale." --seeds 3 -- \
  --steps 4800 --eval_every 1600 --arch hybrid_blend --mixing multi_scale_blend $LS $WC

echo "=== exp-057 tri_scale @9600 (recorde absoluto) ==="
run --id exp-057 --name tri_scale_ls_wc_9600 --verdict INCONCLUSIVA \
  --novelty "NOVEL | escala 9600 do tri_scale." \
  --lesson "Push record beyond 92.98%." --seeds 3 -- \
  --steps 9600 --eval_every 3200 --mixing tri_scale_blend $LS --warmup_steps 200 --scheduler cosine

echo "=== exp-058 HYBRID @7200 ==="
run --id exp-058 --name hybrid_tri_ls_wc_7200 --verdict INCONCLUSIVA \
  --novelty "NOVEL | Hybrid escala 7200." \
  --lesson "Regra 3 hybrid." --seeds 3 -- \
  --steps 7200 --eval_every 2400 --arch hybrid_blend --mixing tri_scale_blend $LS --warmup_steps 150 --scheduler cosine

echo "=== BATCH 7 DONE ==="
