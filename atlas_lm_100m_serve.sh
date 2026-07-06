#!/bin/bash
# Serve dashboard LM 100M em http://localhost:8766
set -e
cd "$(dirname "$0")"
python3 atlas_lm_100m_dashboard.py
echo "Dashboard: http://localhost:8766/lm_100m_dashboard.html"
echo "JSON:      http://localhost:8766/lm_100m_status.json"
python3 -m http.server 8766
