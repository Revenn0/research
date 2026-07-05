#!/usr/bin/env python3
"""Atualiza vereditos e lições no experiments_log.jsonl."""

import json
from pathlib import Path

UPDATES = {
    "exp-014": ("MORTA", "Spatial Gate ReLU: −0.89pp @1200. Gate sem parâmetros suprime demais."),
    "exp-015": ("MORTA", "Norm Feedback: −0.13pp @1200. Feedback global instável, sem ganho."),
    "exp-016": ("PROMISSORA", "Local Blend NOVEL: +0.70pp @1200 vs baseline (90.48±0.38%)."),
    "exp-017": ("MORTA", "Signed Sqrt: colapso total (10% acc, NaN loss). Ativação inviável."),
    "exp-018": ("PROMISSORA", "Local Blend @2400: +0.77pp vs baseline (91.58±0.35%). Ganho cresce com budget."),
    "exp-019": ("INCONCLUSIVA", "Variance Gated ReLU: +0.09pp @1200, dentro da incerteza; lento (~24 sps)."),
}

path = Path("experiments_log.jsonl")
lines = []
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    entry = json.loads(line)
    eid = entry.get("id")
    if eid in UPDATES:
        entry["verdict"], entry["lesson"] = UPDATES[eid]
    lines.append(entry)

path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines) + "\n", encoding="utf-8")
print(f"Atualizados {len(UPDATES)} vereditos.")
