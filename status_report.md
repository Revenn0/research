# ATLAS — Status Report

**Última atualização:** 2026-07-05 14:05 UTC  
**Meta:** 10 PROMISSORAS → **12/10 ATINGIDA** ✓  
**Total experimentos:** 33 (exp-000 a exp-032)

## Campeão geral

**exp-027: local_blend + label_smoothing @2400 = 91.78% (+0.97pp vs baseline)**

## Resumo do ciclo

| Fase | Experimentos | PROMISSORAS encontradas |
|------|-------------|------------------------|
| Calibração (exp-000–013) | 14 | 2 (RECOMB) |
| NOVEL v1 (exp-014–019) | 6 | 2 (local_blend) |
| Mutações (exp-020–032) | 13 | 8 |
| **Total** | **33** | **12** |

## Mecanismos NOVEL promovidos

| Mecanismo | Melhor ID | Acc @2400 | Δ |
|-----------|-----------|-----------|---|
| local_blend | exp-018 | 91.58% | +0.77pp |
| local_blend_k5 | exp-026 | 91.71% | +0.90pp |
| laplacian_blend | exp-028 | 91.58% | +0.77pp |

## Lições finais

1. **Gate por média > variância** (exp-020 MORTA vs exp-016 PROMISSORA)
2. **Kernel 5×5 > 3×3** (+0.14pp vs local_blend @1200)
3. **Label smoothing compõe bem com blend** (+0.19pp sobre blend isolado @1200)
4. **Combos warmup+cosine + blend** não superam blend+LS

## Arquivos

- `promissora_tracker.md` — 12/10
- `promotion_report_exp-027.md` — campeão geral
- `promotion_report_exp-018.md` — campeão NOVEL puro

**Loop pausado:** meta de 10 promissores atingida.
