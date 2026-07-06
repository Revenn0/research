#!/bin/bash
# Atualiza painel LM 100M a cada 60s
set -e
cd "$(dirname "$0")"
while true; do
  python3 atlas_lm_100m_dashboard.py >/dev/null 2>&1 || true
  sleep 60
done
