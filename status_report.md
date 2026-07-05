# ATLAS — Status Report

**Última atualização:** 2026-07-05 12:56 UTC  
**Hardware:** CPU Linux, PyTorch 2.12.1+cpu, ~41 steps/s baseline / ~27 steps/s com local_blend  
**Sessão:** retomada; sem daemon/background ativo  
**Experimento ativo:** ciclo NOVEL — exp-019+ na fila

## Resumo executivo

Regra v2 aplicada: leaderboard re-classificado (NOVEL|RECOMB|CONHECIDA). **Primeiro campeão NOVEL encontrado:** Local Blend Mixing (`exp-016`/`exp-018`), +0.77pp @2400 steps vs baseline, superando também o melhor RECOMB (warmup+cosine).

Total: **19 experimentos reais** (exp-000 a exp-018).

## Re-classificação completa (exp-000 a exp-013)

| Rótulo | IDs |
|--------|-----|
| BASELINE | exp-000, exp-010 |
| CONHECIDA | exp-001 a exp-006, exp-009, exp-012, exp-013 |
| RECOMB | exp-007, exp-008, exp-011 |
| NOVEL | *(nenhum neste lote)* |

## Resultados NOVEL (exp-014 a exp-018)

| ID | Mecanismo | Rótulo | Val Acc @1200 | Val Acc @2400 | Veredito |
|----|-----------|--------|---------------|---------------|----------|
| exp-014 | Spatial Gate ReLU | NOVEL | 88.89% (−0.89pp) | — | MORTA |
| exp-015 | Norm Feedback Optimizer | NOVEL | 89.65% (−0.13pp) | — | MORTA |
| exp-016 | Local Blend Mixing | NOVEL | **90.48% (+0.70pp)** | — | PROMISSORA |
| exp-017 | Signed Sqrt Activation | NOVEL | 10.0% (colapso) | — | MORTA |
| exp-018 | Local Blend (escala) | NOVEL | — | **91.58% (+0.77pp)** | PROMISSORA |

### Teste adversarial — Local Blend (campeão)

- **Mais próximo:** SKNet (Li et al. 2019) — multi-kernel + softmax
- **Defesa:** único depthwise 3×3; gate = σ(spatial_mean(x)); sem branches paralelos
- **Veredito novidade:** NOVEL mantido

## Comparação com calibração

| Campeão | Rótulo | Acc @2400 | Δ baseline |
|---------|--------|-----------|------------|
| **exp-018 Local Blend** | **NOVEL** | **91.58%** | **+0.77pp** |
| exp-011 warmup+cosine | RECOMB | 91.38% | +0.57pp |
| exp-010 baseline | BASELINE | 90.81% | — |

## Lições do ciclo NOVEL

1. **Gates sem parâmetros** (exp-014) podem suprimir demais — energia relativa ≠ SE-Net.
2. **Feedback global de norma** (exp-015) não substitui adaptação per-param do Adam.
3. **Signed sqrt** (exp-017) causa NaN — compressão agressiva destrói gradientes.
4. **Local blend com init dirac** (exp-016) começa ≈identidade e aprende mixing gradual — ganho robusto.

## Próximos experimentos (fila NOVEL)

- [ ] exp-019: Variance-Gated ReLU — `relu(x) * sigmoid(var_spatial(x))`
- [ ] exp-020: Local Blend + gate de variância (mutação do campeão)
- [ ] exp-021: Local Blend @4800 steps (re-teste escala 2× adicional)
- [ ] Ablation: local_blend sem dirac init

## Arquivos de estado

Total: **20 experimentos** (exp-000 a exp-019).
- `leaderboard.md` — re-classificado v2
- `promotion_report_exp-018.md` — campeão NOVEL para GPU
