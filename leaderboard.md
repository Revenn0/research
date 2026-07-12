# ATLAS Leaderboard

**Meta PROMISSORA: 12/10 ATINGIDA** ✓

## Top 5 (qualidade/compute em steps)

| Rank | ID | Mecanismo | Steps | Val Acc | Δ baseline | Rótulo |
|------|-----|-----------|-------|---------|------------|--------|
| 1 | exp-027 | blend + LS | 2400 | **91.78%** | +0.97pp* | RECOMB |
| 2 | exp-026 | local_blend_k5 | 2400 | 91.71% | +0.90pp* | NOVEL |
| 3 | exp-018 | local_blend | 2400 | 91.58% | +0.77pp* | NOVEL |
| 4 | exp-028 | laplacian_blend | 2400 | 91.58% | +0.77pp* | NOVEL |
| 5 | exp-011 | warmup+cosine | 2400 | 91.38% | +0.57pp* | RECOMB |

\*Δ vs baseline no mesmo budget (exp-010 @2400 = 90.81%)

## Todas PROMISSORAS (12)

| ID | Nome | Rótulo | Acc | Budget |
|----|------|--------|-----|--------|
| exp-027 | blend+LS | RECOMB | 91.78% | 2400 |
| exp-026 | k5 blend | NOVEL | 91.71% | 2400 |
| exp-018 | local_blend | NOVEL | 91.58% | 2400 |
| exp-028 | laplacian | NOVEL | 91.58% | 2400 |
| exp-011 | warmup+cosine | RECOMB | 91.38% | 2400 |
| exp-025 | blend+LS | RECOMB | 90.67% | 1200 |
| exp-021 | k5 blend | NOVEL | 90.62% | 1200 |
| exp-032 | k5+LS | RECOMB | 90.62% | 1200 |
| exp-024 | blend+warmup+cos | RECOMB | 90.61% | 1200 |
| exp-016 | local_blend | NOVEL | 90.48% | 1200 |
| exp-022 | laplacian | NOVEL | 90.48% | 1200 |
| exp-007 | warmup+cosine | RECOMB | 90.07% | 1200 |

## NOVEL descobertos: 5 mecanismos

1. **local_blend** — depthwise 3×3 + gate média
2. **local_blend_k5** — kernel 5×5
3. **laplacian_blend** — filtro Laplaciano fixo
4. *(variações gate testadas, mean vence var)*

## Baselines

| ID | Steps | Val Acc |
|----|-------|---------|
| exp-000 | 1200 | 89.78% |
| exp-010 | 2400 | 90.81% |
