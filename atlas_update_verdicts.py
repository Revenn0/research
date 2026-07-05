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
    "exp-020": ("MORTA", "local_blend_var: −0.07pp. Gate por variância pior que média."),
    "exp-021": ("PROMISSORA", "local_blend_k5: +0.84pp @1200 (90.62±0.24%). Melhor mutação blend."),
    "exp-022": ("PROMISSORA", "laplacian_blend: +0.70pp @1200. Filtro Laplaciano NOVEL."),
    "exp-023": ("INCONCLUSIVA", "dual_gate_blend: +0.53pp @1200. Ganho marginal."),
    "exp-024": ("PROMISSORA", "local_blend+warmup+cosine: +0.83pp @1200 RECOMB."),
    "exp-025": ("PROMISSORA", "local_blend+LS: +0.89pp @1200. Melhor RECOMB @1200."),
    "exp-026": ("PROMISSORA", "local_blend_k5 @2400: 91.71% (+0.90pp vs baseline 2400). Escala validada."),
    "exp-027": ("PROMISSORA", "blend+LS @2400: 91.78% (+0.97pp). CAMPEÃO GERAL."),
    "exp-028": ("PROMISSORA", "laplacian_blend @2400: 91.58% (+0.77pp). Escala validada."),
    "exp-029": ("INCONCLUSIVA", "rms_free: +0.28pp @1200, dentro da incerteza."),
    "exp-030": ("MORTA", "grad_shrink: −0.03pp @1200."),
    "exp-031": ("INCONCLUSIVA", "k5+warmup+cosine: +0.69pp, não supera k5+LS."),
    "exp-032": ("PROMISSORA", "k5+LS @1200: +0.84pp, equivale exp-021."),
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
