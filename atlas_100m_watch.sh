#!/bin/bash
# Atualiza painel 100M a cada 30s + heartbeat legível no desktop
set -e
cd "$(dirname "$0")"
HEARTBEAT="batch_100m_live.txt"
while true; do
  python3 atlas_100m_dashboard.py >/dev/null 2>&1 || true
  {
    echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    if pgrep -f 'atlas_batch_100m.sh' >/dev/null; then echo "BATCH: RODANDO"; else echo "BATCH: PARADO"; fi
    TRAIN_PID=$(pgrep -f 'train_baseline.py.*width_mult' | head -1)
    if [ -n "$TRAIN_PID" ]; then
      ps -p "$TRAIN_PID" -o etime=,pcpu=,pmem=,rss= 2>/dev/null | awk '{print "TREINO PID '"$TRAIN_PID"': tempo="$1" CPU="$2"% RAM="$3"% RSS="$4"KB"}'
      tr '\0' ' ' < /proc/$TRAIN_PID/cmdline 2>/dev/null | grep -oP 'seed \d+' || true
    else
      echo "TREINO: nenhum processo ativo"
    fi
    python3 -c "import json;d=json.load(open('promissora_100m_status.json'));print(f\"EXP: {d.get('current_experiment_id')} seed {d.get('current_seed')} | done {d['progress']['done']}/{d['progress']['total']}\")" 2>/dev/null || true
    echo ""
  } > "$HEARTBEAT"
  sleep 30
done
