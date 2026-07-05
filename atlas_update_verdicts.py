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
    "exp-033": ("PROMISSORA", "k5+LS @2400: 91.80% — supera top-5 anterior (+0.02pp vs exp-027)."),
    "exp-034": ("INCONCLUSIVA", "multi_scale+LS @2400: 91.69%, não supera top-5."),
    "exp-035": ("PROMISSORA", "multi_scale @2400: 91.59%, supera exp-018/028 marginalmente."),
    "exp-036": ("INCONCLUSIVA", "cosine_gate+LS @2400: 91.66%, top-3 mas não top-1."),
    "exp-037": ("MORTA", "entropy_gate+LS @2400: 91.26%, abaixo top-5."),
    "exp-038": ("MORTA", "post_act+LS @2400: 91.43%, abaixo top-5."),
    "exp-039": ("MORTA", "k5+LS+wc @2400: 91.57%, warmup não ajuda com k5+LS."),
    "exp-040": ("PROMISSORA", "multi_scale+LS+wc @2400: 91.80%, empata/supera top-5."),
    "exp-041": ("PROMISSORA", "k5+LS @4800: 92.15% (+0.53pp vs baseline 4800)."),
    "exp-042": ("REVOLUCIONARIA", "multi_scale+LS+wc @4800: 92.68% (+1.06pp). NOVO CAMPEÃO."),
    "exp-043": ("MORTA", "k5+LS h256 @2400: 91.58%, hidden maior não ajuda."),
    "exp-044": ("INCONCLUSIVA", "cascade_blend+LS @2400: 91.72%, não supera top-5."),
    "exp-045": ("INCONCLUSIVA", "k5+LS dropout0.05 @2400: 91.69%, não supera top-5."),
    "exp-046": ("BASELINE", "Baseline @4800: 91.62±0.30%."),
    "exp-047": ("REVOLUCIONARIA", "multi_scale+LS+wc @7200: 92.90% (+1.28pp vs baseline 4800). Ganho cresce."),
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
