#!/bin/bash
# ============================================================================
# ATLAS H100 — teste unitário em escala das 12 PROMISSORAS vs baselines maiores
#
# Requisitos: GPU NVIDIA (H100), PyTorch com CUDA.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#
# Uso:  ./atlas_h100_batch.sh 2>&1 | tee batch_h100.log
#
# Fases:
#   0. Testes unitários dos mecanismos (pytest-style, ~1 min)
#   A. 12 promissoras + baseline @ width×4 (128/256 ch), 24000 steps, batch 512
#   B. top-4 + baseline @ width×8 (256/512 ch), 48000 steps, batch 1024
#
# Comparação justa: cada mecanismo vs baseline NA MESMA escala/budget.
# Resultados em experiments_log.jsonl (ids exp-h100-*).
# ============================================================================
set -e
cd "$(dirname "$0")"

echo "========== GPU CHECK =========="
python3 - << 'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERRO: CUDA indisponível. Este script requer GPU (H100).")
    print("Nesta máquina rode apenas os testes unitários: python3 tests/test_promissora_mechanisms.py")
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
EOF

run() { python3 atlas_run.py "$@"; }

# Escala A: modelo 4x mais largo, 10x mais steps que o teste CPU original
A="--steps 24000 --eval_every 4000 --batch_size 512 --lr 0.002 --width_mult 4.0 --hidden_dim 512 --amp --num_workers 8 --device cuda"
# Escala B: modelo 8x mais largo, budget dobrado
B="--steps 48000 --eval_every 8000 --batch_size 1024 --lr 0.003 --width_mult 8.0 --hidden_dim 1024 --amp --num_workers 8 --device cuda"
WC="--warmup_steps 1000 --scheduler cosine"
LS="--label_smoothing 0.1"

echo "========== FASE 0: TESTES UNITÁRIOS =========="
python3 tests/test_promissora_mechanisms.py

echo "========== FASE A: 12 PROMISSORAS + BASELINE @ width×4, 24k steps =========="

echo "=== exp-h100-baseline-A | baseline maior (referência) ==="
run --id exp-h100-baseline-A --name h100_baseline_w4_24k --verdict INCONCLUSIVA \
  --novelty "BASELINE | width×4 @24k — régua da fase A." \
  --lesson "Baseline maior H100." --seeds 3 -- $A

echo "=== exp-h100-01 | p007/011 warmup+cosine ==="
run --id exp-h100-01 --name h100_warmup_cosine_w4 --verdict INCONCLUSIVA \
  --novelty "RECOMB | warmup+cosine em escala H100." \
  --lesson "Unit test escala: warmup+cosine." --seeds 3 -- $A $WC

echo "=== exp-h100-02 | p016/018 local_blend ==="
run --id exp-h100-02 --name h100_local_blend_w4 --verdict INCONCLUSIVA \
  --novelty "NOVEL | local_blend em escala H100." \
  --lesson "Unit test escala: local_blend." --seeds 3 -- $A --mixing local_blend

echo "=== exp-h100-03 | p021/026 local_blend_k5 ==="
run --id exp-h100-03 --name h100_k5_w4 --verdict INCONCLUSIVA \
  --novelty "NOVEL | k5 blend em escala H100." \
  --lesson "Unit test escala: k5." --seeds 3 -- $A --mixing local_blend_k5

echo "=== exp-h100-04 | p022/028 laplacian_blend ==="
run --id exp-h100-04 --name h100_laplacian_w4 --verdict INCONCLUSIVA \
  --novelty "NOVEL | laplacian blend em escala H100." \
  --lesson "Unit test escala: laplacian." --seeds 3 -- $A --mixing laplacian_blend

echo "=== exp-h100-05 | p024 blend+warmup+cosine ==="
run --id exp-h100-05 --name h100_blend_wc_w4 --verdict INCONCLUSIVA \
  --novelty "RECOMB | blend+wc em escala H100." \
  --lesson "Unit test escala: blend+wc." --seeds 3 -- $A --mixing local_blend $WC

echo "=== exp-h100-06 | p025/027 blend+LS (ex-campeão) ==="
run --id exp-h100-06 --name h100_blend_ls_w4 --verdict INCONCLUSIVA \
  --novelty "RECOMB | blend+LS em escala H100." \
  --lesson "Unit test escala: blend+LS." --seeds 3 -- $A --mixing local_blend $LS

echo "=== exp-h100-07 | p032 k5+LS ==="
run --id exp-h100-07 --name h100_k5_ls_w4 --verdict INCONCLUSIVA \
  --novelty "RECOMB | k5+LS em escala H100." \
  --lesson "Unit test escala: k5+LS." --seeds 3 -- $A --mixing local_blend_k5 $LS

echo "=== exp-h100-08 | stack completo blend+LS+wc ==="
run --id exp-h100-08 --name h100_blend_ls_wc_w4 --verdict INCONCLUSIVA \
  --novelty "RECOMB | stack completo das promissoras." \
  --lesson "Unit test escala: stack blend+LS+wc." --seeds 3 -- $A --mixing local_blend $LS $WC

echo "=== exp-h100-09 | campeões atuais: multi_scale+LS+wc ==="
run --id exp-h100-09 --name h100_multi_scale_ls_wc_w4 --verdict INCONCLUSIVA \
  --novelty "NOVEL | campeão multi_scale em escala H100." \
  --lesson "Campeão multi_scale @H100." --seeds 3 -- $A --mixing multi_scale_blend $LS $WC

echo "=== exp-h100-10 | recorde: tri_scale+LS+wc ==="
run --id exp-h100-10 --name h100_tri_scale_ls_wc_w4 --verdict INCONCLUSIVA \
  --novelty "NOVEL | recordista tri_scale em escala H100." \
  --lesson "Recorde tri_scale @H100." --seeds 3 -- $A --mixing tri_scale_blend $LS $WC

echo "========== FASE B: TOP-4 + BASELINE @ width×8, 48k steps =========="

echo "=== exp-h100-baseline-B | baseline gigante ==="
run --id exp-h100-baseline-B --name h100_baseline_w8_48k --verdict INCONCLUSIVA \
  --novelty "BASELINE | width×8 @48k — régua da fase B." \
  --lesson "Baseline gigante H100." --seeds 3 -- $B

echo "=== exp-h100-B1 | blend+LS ==="
run --id exp-h100-B1 --name h100_blend_ls_w8 --verdict INCONCLUSIVA \
  --novelty "RECOMB | blend+LS width×8." \
  --lesson "Fase B: blend+LS." --seeds 3 -- $B --mixing local_blend $LS

echo "=== exp-h100-B2 | k5+LS ==="
run --id exp-h100-B2 --name h100_k5_ls_w8 --verdict INCONCLUSIVA \
  --novelty "RECOMB | k5+LS width×8." \
  --lesson "Fase B: k5+LS." --seeds 3 -- $B --mixing local_blend_k5 $LS

echo "=== exp-h100-B3 | multi_scale+LS+wc ==="
run --id exp-h100-B3 --name h100_multi_scale_w8 --verdict INCONCLUSIVA \
  --novelty "NOVEL | multi_scale width×8." \
  --lesson "Fase B: multi_scale." --seeds 3 -- $B --mixing multi_scale_blend $LS $WC

echo "=== exp-h100-B4 | tri_scale+LS+wc ==="
run --id exp-h100-B4 --name h100_tri_scale_w8 --verdict INCONCLUSIVA \
  --novelty "NOVEL | tri_scale width×8." \
  --lesson "Fase B: tri_scale." --seeds 3 -- $B --mixing tri_scale_blend $LS $WC

echo "========== H100 BATCH DONE =========="
echo "Ranking final:"
python3 - << 'EOF'
import json
rows = []
for line in open("experiments_log.jsonl"):
    e = json.loads(line)
    if e["id"].startswith("exp-h100"):
        r = e["result"]
        rows.append((r["best_val_acc_mean"], e["id"], e["name"], r["best_val_acc_std"]))
rows.sort(reverse=True)
for acc, eid, name, std in rows:
    print(f"{acc*100:6.2f}% ± {std*100:.2f}%  {eid:22} {name}")
EOF
