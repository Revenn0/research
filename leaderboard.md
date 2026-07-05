# ATLAS Leaderboard

Ranking por **best_val_acc_mean** em comparações justas (mesmo budget de steps, mesmas seeds 1000–1002).

## Tier S — PROMISSORA (validar em GPU)

| Rank | ID | Nome | Steps | Val Acc | Δ baseline | Notas |
|------|-----|------|-------|---------|------------|-------|
| 1 | exp-011 | warmup_cosine_2400 | 2400 | **91.38±0.14%** | +0.57pp | Ganho cresce com budget |
| 2 | exp-007 | warmup100_cosine | 1200 | **90.07±0.32%** | +0.29pp | Mesmo mecanismo, budget menor |

## Tier B — INCONCLUSIVA (marginal / alta incerteza)

| Rank | ID | Nome | Steps | Val Acc | Δ baseline |
|------|-----|------|-------|---------|------------|
| 3 | exp-013 | ls_2400steps | 2400 | 90.89±0.07% | +0.08pp |
| 4 | exp-002 | label_smoothing_0.1 | 1200 | 89.98±0.18% | +0.20pp |
| 5 | exp-004 | mixup_0.2 | 1200 | 89.81±0.37% | +0.03pp |
| 6 | exp-012 | grad_centralize | 1200 | 89.66±0.18% | -0.12pp |

## Tier F — MORTA

| ID | Nome | Val Acc | Δ baseline | Motivo |
|----|------|---------|------------|--------|
| exp-003 | ema_0.999 | 88.25% | -1.53pp | EMA lento demais |
| exp-006 | sqrt_lr_decay | 89.19% | -0.59pp | Schedule inferior |
| exp-009 | cutout_8 | 89.53% | -0.25pp | Augmentation agressiva demais |
| exp-005 | batchnorm | 89.58% | -0.20pp | Lento + sem ganho |
| exp-001 | adamw_cosine | 89.76% | -0.02pp | Neutro/negativo |
| exp-008 | ls_warmup_cosine_combo | 89.95% | +0.17pp | Não supera exp-007 |

## Baselines de referência

| ID | Steps | Val Acc |
|----|-------|---------|
| exp-000 | 1200 | 89.78±0.31% |
| exp-010 | 2400 | 90.81±0.31% |

## Eficiência compute (qualidade / tempo)

| ID | Val Acc | steps/s | Acc·s⁻¹ ×10³ |
|----|---------|---------|--------------|
| exp-007 | 90.07% | 40.8 | 3.67 |
| exp-000 | 89.78% | 41.7 | 3.74 |
| exp-005 | 89.58% | 29.0 | 2.60 |

Baseline Adam fixo ainda é ligeiramente mais eficiente em acc/s, mas exp-007 entrega mais acurácia final pelo mesmo número de steps — **melhor qualidade/compute em termos de steps**.
