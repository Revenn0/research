#!/bin/bash
# Verificação independente ATL-0032 — todas as tarefas do prompt
set -e
cd /workspace
LOG=atlas/verify_0032.log
exec > >(tee -a "$LOG") 2>&1

SEEDS=(42 1337 7)
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

run_baseline() {
  local tag=$1 seed=$2 steps=$3 eval=$4 extra=${5:-}
  echo "=== BASELINE seed=$seed steps=$steps start=$(ts) ==="
  t0=$(date +%s)
  python3 atlas/train_baseline.py --tag "$tag" --seed "$seed" \
    --total_steps "$steps" --eval_every "$eval" \
    --n_layer 3 --n_head 4 --n_embd 64 --block_size 64 --batch 16 --dropout 0.1 \
    $extra | python3 -c "import json,sys; d=json.load(sys.stdin); print('val_loss', d['final_val_loss'], 'wall', d['wall_time_s'])"
  echo "end=$(ts) elapsed=$(( $(date +%s) - t0 ))s"
}

run_gene() {
  local tag=$1 seed=$2 steps=$3 eval=$4 extra=$5
  echo "=== GENE $tag seed=$seed steps=$steps start=$(ts) ==="
  t0=$(date +%s)
  python3 atlas/train_gene.py --tag "$tag" --seed "$seed" \
    --total_steps "$steps" --eval_every "$eval" \
    --n_layer 3 --n_head 4 --n_embd 64 --block_size 64 --batch 16 --dropout 0.1 \
    $extra | python3 -c "import json,sys; d=json.load(sys.stdin); print('val_loss', d['final_val_loss'], 'wall', d['wall_time_s'])"
  echo "end=$(ts) elapsed=$(( $(date +%s) - t0 ))s"
}

echo "========== TAREFA 3: BUG CHECKS =========="
python3 atlas/verify_bugs.py

echo "========== TAREFA 1: BASELINE 3 seeds @8000 =========="
for s in "${SEEDS[@]}"; do
  run_baseline baseline-v3 "$s" 8000 1000
done

echo "========== TAREFA 2: ATL-0032 3 seeds @2760 =========="
for s in "${SEEDS[@]}"; do
  run_gene ATL-0032-timequal "$s" 2760 460 \
    "--mixing long-conv-fft --norm layernorm --activation gelu --optimizer signlion --schedule cosine-warmup"
done

echo "========== TAREFA 4: ABLAÇÃO attn+signlion @8000 =========="
for s in "${SEEDS[@]}"; do
  run_gene ablation-attn-signlion "$s" 8000 1000 \
    "--mixing global-attn --norm layernorm --activation gelu --optimizer signlion --schedule cosine-warmup"
done

echo "========== TAREFA 5: ABLAÇÃO conv+adamw @2760 =========="
for s in "${SEEDS[@]}"; do
  run_gene ablation-conv-adamw "$s" 2760 460 \
    "--mixing long-conv-fft --norm layernorm --activation gelu --optimizer adamw --schedule cosine-warmup"
done

echo "========== TAREFA 6: block_size=256 time-equal =========="
for s in "${SEEDS[@]}"; do
  run_baseline baseline-b256 "$s" 500 250 "--block_size 256"
done
for s in "${SEEDS[@]}"; do
  run_gene ATL-0032-b256 "$s" 250 125 \
    "--block_size 256 --mixing long-conv-fft --norm layernorm --activation gelu --optimizer signlion --schedule cosine-warmup"
done

echo "========== ATL-0032 VERIFICATION DONE $(ts) =========="
