#!/usr/bin/env python3
"""Atualiza vereditos e lições no experiments_log.jsonl."""

import json
from pathlib import Path

UPDATES = {
    "exp-000": ("BASELINE", "Referência: 89.78±0.31% @1200 steps. Bem acima do acaso (10%)."),
    "exp-001": ("MORTA", "AdamW+cosine não supera baseline neste regime CPU/CNN pequena."),
    "exp-002": ("INCONCLUSIVA", "+0.20pp @1200; marginal @2400 (+0.08pp). Ganho dentro da incerteza."),
    "exp-003": ("MORTA", "EMA 0.999 prejudica (-1.5pp): decay alto demais para 1200 steps."),
    "exp-004": ("INCONCLUSIVA", "Mixup 0.2 neutro (+0.03pp), alta variância entre seeds."),
    "exp-005": ("MORTA", "BatchNorm: -0.2pp acurácia e -30% throughput — pior custo-benefício."),
    "exp-006": ("MORTA", "Sqrt LR decay inferior ao baseline fixo (-0.6pp)."),
    "exp-007": ("PROMISSORA", "Warmup100+cosine: +0.29pp @1200 steps vs baseline."),
    "exp-008": ("MORTA", "Combo LS+warmup+cosine não supera warmup+cosine isolado."),
    "exp-009": ("MORTA", "Cutout 8 piora (-0.45pp) neste CNN pequeno."),
    "exp-010": ("BASELINE", "Referência escalada: 90.81±0.31% @2400 steps."),
    "exp-011": ("PROMISSORA", "Warmup+cosine mantém +0.57pp @2400 — ganho cresce com budget."),
    "exp-012": ("MORTA", "Gradient centralization: -0.15pp, sem benefício aqui."),
    "exp-013": ("INCONCLUSIVA", "Label smoothing @2400: +0.08pp, dentro da incerteza."),
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
