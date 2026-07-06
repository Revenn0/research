# Relatório — CIFAR-10 v2 (régua justa baseline_ls_wc)

**Data:** 2026-07-06 | **GPU:** A100 40GB | **STEPS:** 400 | **3 seeds**

## Ranking (régua justa)

| Rank | Método | Acc | Δ baseline_ls_wc | Veredito |
|------|--------|-----|------------------|----------|
| 1 | **multi_scale_ls_wc** | **71.06%** | **+3.48 pp** | **CONFIRMADA** |
| 2 | tri_scale_ls_wc | 71.04% | +3.45 pp | CONFIRMADA |
| 3 | blend_ls_wc | 70.96% | +3.38 pp | CONFIRMADA |
| 4 | se_ls_wc (melhor LIT) | 69.44% | +1.86 pp | CONFIRMADA_LIT |
| 5 | cbam_ls_wc | 69.14% | +1.55 pp | CONFIRMADA_LIT |
| 6 | **baseline_ls_wc** | **67.58%** | 0 | régua |
| 7 | cbam | 66.13% | −1.45 pp | REFUTADA |
| 8 | se_block | 61.02% | −6.57 pp | REFUTADA |

**Baseline:** 67.58% ± 0.55% (estável — comparável aos outros stacks)

## Head-to-head principal

| | Acc |
|---|-----|
| Melhor NOVEL (multi_scale) | **71.06%** |
| Melhor LITERATURA (se_ls_wc) | 69.44% |
| **Δ NOVEL vs LIT** | **+1.62 pp** |

## Veredito ATLAS: **CONFIRMADA**

Generalização além do FashionMNIST **confirmada** com régua justa.

## Comparação v1 → v2

| | v1 (baseline cru) | v2 (baseline_ls_wc) |
|---|-------------------|---------------------|
| Baseline | 33% ± 20% ❌ | 67.58% ± 0.55% ✅ |
| Δ NOVEL vs LIT | +1.62 pp | +1.62 pp (igual) |
| Confiabilidade | baixa | **alta** |

## Implicações

1. **multi/tri_scale** confirmados em dataset mais difícil
2. **Stack LS+wc** essencial (SE/CBAM puros falham sem stack)
3. NOVEL bate literatura publicada com **mesma receita de treino**
4. Próximo passo: **LM 100M** com multi/tri_scale TokenBlend
