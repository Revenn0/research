#!/bin/bash
# ============================================================================
# ATLAS batch 100M — execução PARALELA dos experimentos pendentes
#
# Uso:
#   MAX_PARALLEL=2 ./atlas_batch_100m_parallel.sh 2>&1 | tee -a batch_100m_parallel.log
#
# MAX_PARALLEL: jobs simultâneos (default auto ~2 em VM 15GB)
# MAX_PARALLEL=9 força todos pendentes (risco OOM se RAM < ~60GB)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

done_id() {
  python3 -c "
import json, sys
done = {json.loads(l)['id'] for l in open('experiments_log.jsonl') if l.strip() and 'exp-100m' in l}
print('yes' if sys.argv[1] in done else 'no')
" "$1"
}

auto_max_parallel() {
  python3 - << 'EOF'
try:
    with open("/proc/meminfo") as f:
        mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in f if ":" in line)}
    avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
    jobs = max(1, min(9, int(avail_kb / (6.5 * 1024 * 1024))))
except OSError:
    jobs = 2
print(jobs)
EOF
}

MAX_PARALLEL="${MAX_PARALLEL:-$(auto_max_parallel)}"
SCALE_BASE="--width_mult 16.0 --hidden_dim 2048 --batch_size 64 --lr 0.001 --weight_decay 1e-4 --device cpu"
S1000="--steps 1000 --eval_every 250 --log_every 200"
WC="--warmup_steps 100 --scheduler cosine"
LS="--label_smoothing 0.1"

if [ "$MAX_PARALLEL" -gt 1 ]; then
  NW="--num_workers 0"
else
  NW="--num_workers 2"
fi
SCALE="$SCALE_BASE $NW"

LOG_DIR="results/100m/logs"
mkdir -p "$LOG_DIR"

run_exp() {
  local id="$1"
  shift
  if [ "$(done_id "$id")" = "yes" ]; then
    echo "[$(date -u '+%H:%M:%S')] SKIP $id (já concluído)"
    return 0
  fi
  echo "[$(date -u '+%H:%M:%S')] START $id"
  python3 atlas_run.py \
    --result_dir results/100m \
    --no_refresh_dashboard \
    "$@" \
    > "$LOG_DIR/${id}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date -u '+%H:%M:%S')] DONE  $id"
  else
    echo "[$(date -u '+%H:%M:%S')] FAIL  $id (exit $rc)" >&2
    return $rc
  fi
  python3 atlas_100m_dashboard.py >/dev/null 2>&1 || true
}

launch() {
  local id="$1"
  shift
  if [ "$(done_id "$id")" = "yes" ]; then
    echo "=== SKIP $id (já concluído) ==="
    return 0
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n 2>/dev/null || wait || true
    python3 atlas_100m_dashboard.py >/dev/null 2>&1 || true
  done
  run_exp "$id" "$@" &
  echo "=== LAUNCHED $id (running=$(jobs -rp | wc -l)/$MAX_PARALLEL) ==="
  python3 atlas_100m_dashboard.py >/dev/null 2>&1 || true
}

echo "========== BATCH 100M PARALELO (MAX_PARALLEL=$MAX_PARALLEL) =========="
free -h | head -2

launch exp-100m-baseline \
  --id exp-100m-baseline --name baseline_100m --verdict INCONCLUSIVA \
  --novelty "BASELINE | referência @100M." --lesson "Baseline escala 100M." --seeds 3 -- \
  $SCALE $S1000

launch exp-100m-004 \
  --id exp-100m-004 --name p018_blend_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-018 @100M." --lesson "Promissora @100M: local_blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend

launch exp-100m-005 \
  --id exp-100m-005 --name p021_k5_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-021 @100M." --lesson "Promissora @100M: k5 blend." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5

launch exp-100m-006 \
  --id exp-100m-006 --name p022_laplacian_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-022 @100M." --lesson "Promissora @100M: laplacian." --seeds 3 -- \
  $SCALE $S1000 --mixing laplacian_blend

launch exp-100m-007 \
  --id exp-100m-007 --name p024_blend_wc_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-024 @100M." --lesson "Promissora @100M: blend+wc." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $WC

launch exp-100m-008 \
  --id exp-100m-008 --name p025_blend_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-025 @100M." --lesson "Promissora @100M: blend+LS." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend $LS

launch exp-100m-009 \
  --id exp-100m-009 --name p026_k5_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-026 @100M." --lesson "Promissora @100M: k5." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5

launch exp-100m-011 \
  --id exp-100m-011 --name p028_laplacian_100m --verdict INCONCLUSIVA \
  --novelty "NOVEL | promissora exp-028 @100M." --lesson "Promissora @100M: laplacian." --seeds 3 -- \
  $SCALE $S1000 --mixing laplacian_blend

launch exp-100m-012 \
  --id exp-100m-012 --name p032_k5_ls_100m --verdict INCONCLUSIVA \
  --novelty "RECOMB | promissora exp-032 @100M." --lesson "Promissora @100M: k5+LS." --seeds 3 -- \
  $SCALE $S1000 --mixing local_blend_k5 $LS

echo "========== AGUARDANDO JOBS RESTANTES =========="
FAIL=0
while jobs -rp | grep -q .; do
  wait -n 2>/dev/null || { wait || FAIL=$((FAIL + 1)); }
  python3 atlas_100m_dashboard.py >/dev/null 2>&1 || true
done

python3 atlas_100m_dashboard.py
echo "========== BATCH PARALELO CONCLUÍDO (falhas=$FAIL) =========="
