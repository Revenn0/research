# ATLAS — Tracker 100M (12 Promissoras)

**Atualizado:** 2026-07-06T16:08:00.166762+00:00
**Progresso global:** 41.4%
**Concluídos:** 5/13
**Rodando:** exp-100m-006 seed 1001 step 0/1000

| # | ID | Mecanismo | % | Status | Val Acc | Δ base | Veredito |
|---|-----|-----------|---|--------|---------|--------|----------|
| 1 | exp-100m-001 | warmup+cosine | **100.0%** | done | 89.78% | — | INCONCLUSIVA |
| 2 | exp-100m-002 | warmup+cosine | **100.0%** | done | 89.93% | — | INCONCLUSIVA |
| 3 | exp-100m-003 | local_blend | **100.0%** | done | 89.29% | — | INCONCLUSIVA |
| 4 | exp-100m-004 | local_blend | **100.0%** | done | 88.99% | — | INCONCLUSIVA |
| 5 | exp-100m-005 | local_blend_k5 | **100.0%** | done | 89.03% | — | INCONCLUSIVA |
| 6 | exp-100m-006 | laplacian_blend | **38.3%** | running | 89.15% | — | rodando |
| 7 | exp-100m-007 | blend+warmup+cosine | **0.0%** | pending | — | — | pendente |
| 8 | exp-100m-008 | blend+label_smooth | **0.0%** | pending | — | — | pendente |
| 9 | exp-100m-009 | local_blend_k5 | **0.0%** | pending | — | — | pendente |
| 10 | exp-100m-010 | blend+LS (ex-campeão) | **0.0%** | pending | — | — | pendente |
| 11 | exp-100m-011 | laplacian_blend | **0.0%** | pending | — | — | pendente |
| 12 | exp-100m-012 | k5+label_smooth | **0.0%** | pending | — | — | pendente |
| 13 | exp-100m-baseline | baseline (referência) | **0.0%** | pending | — | — | pendente |

Critérios: `verdict_criteria_100m.md`

| Veredito | Δ @100M |
|----------|---------|
| REFUTADA | ≤ −0,50 pp |
| INCONCLUSIVA | −0,49 a +0,29 pp |
| PROMISSORA | +0,30 a +0,79 pp |
| REVOLUCIONÁRIA | ≥ +0,80 pp (std ≤ 0,30) |
