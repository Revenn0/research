#!/bin/bash
# Atualiza painel 100M a cada 30s (rode em tmux junto ao batch)
set -e
cd "$(dirname "$0")"
while true; do
  python3 atlas_100m_dashboard.py >/dev/null 2>&1 || true
  sleep 30
done
