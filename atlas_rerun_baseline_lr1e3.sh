#!/bin/bash
# Re-run baseline @ lr=1e-3 (calibrado p/ val~1.998 reportado)
set -e
cd /workspace
for s in 42 1337 7; do
  echo "=== baseline lr=1e-3 seed=$s $(date -u +%H:%M:%S) ==="
  python3 atlas/train_baseline.py --tag baseline-v3-lr1e3 --seed $s \
    --total_steps 8000 --eval_every 1000 --lr 0.001 \
    --n_layer 3 --n_head 4 --n_embd 64 --block_size 64 --batch 16 --dropout 0.1 \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['final_val_loss'])"
done
