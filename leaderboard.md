# ATLAS Leaderboard

**Regra v2:** alvo da missão = **NOVEL**. RECOMB e CONHECIDA = calibração.

Ranking por **best_val_acc_mean** (mesmo budget, seeds 1000–1002).

---

## NOVEL — alvo da missão

| Rank | ID | Nome | Steps | Val Acc | Δ baseline | Veredito |
|------|-----|------|-------|---------|------------|----------|
| 1 | exp-018 | local_blend_2400 | 2400 | **91.58±0.35%** | **+0.77pp** | PROMISSORA |
| 2 | exp-016 | local_blend | 1200 | **90.48±0.38%** | **+0.70pp** | PROMISSORA |
| — | exp-014 | spatial_gate_relu | 1200 | 88.89±0.06% | −0.89pp | MORTA |
| — | exp-015 | norm_feedback | 1200 | 89.65±0.23% | −0.13pp | MORTA |
| — | exp-019 | variance_gated_relu | 1200 | 89.87±0.31% | +0.09pp | INCONCLUSIVA |
| — | exp-017 | signed_sqrt | 1200 | 10.00±0.00% | −79.78pp | MORTA |

**Campeão NOVEL:** exp-018 Local Blend — supera também o melhor RECOMB (exp-011: 91.38%).

---

## Calibração — RECOMB

| ID | Nome | Steps | Val Acc | Δ baseline |
|----|------|-------|---------|------------|
| exp-011 | warmup_cosine_2400 | 2400 | 91.38±0.14% | +0.57pp |
| exp-007 | warmup100_cosine | 1200 | 90.07±0.32% | +0.29pp |
| exp-008 | ls_warmup_cosine_combo | 1200 | 89.95±0.21% | +0.17pp |

---

## Calibração — CONHECIDA

| ID | Nome | Steps | Val Acc | Δ baseline |
|----|------|-------|---------|------------|
| exp-013 | ls_2400steps | 2400 | 90.89±0.07% | +0.08pp |
| exp-002 | label_smoothing_0.1 | 1200 | 89.98±0.18% | +0.20pp |
| exp-004 | mixup_0.2 | 1200 | 89.81±0.37% | +0.03pp |
| exp-012 | grad_centralize | 1200 | 89.66±0.18% | −0.12pp |
| exp-001 | adamw_cosine | 1200 | 89.76±0.27% | −0.02pp |
| exp-005 | batchnorm | 1200 | 89.58±0.67% | −0.20pp |
| exp-009 | cutout_8 | 1200 | 89.53±0.23% | −0.25pp |
| exp-006 | sqrt_lr_decay | 1200 | 89.19±0.43% | −0.59pp |
| exp-003 | ema_0.999 | 1200 | 88.25±0.07% | −1.53pp |

---

## Baselines

| ID | Steps | Val Acc |
|----|-------|---------|
| exp-000 | 1200 | 89.78±0.31% |
| exp-010 | 2400 | 90.81±0.31% |

---

## Eficiência compute

| ID | Rótulo | Val Acc | steps/s | Acc por step |
|----|--------|---------|---------|--------------|
| exp-018 | NOVEL | 91.58% | 27.4 | **melhor** |
| exp-011 | RECOMB | 91.38% | 40.9 | bom |
| exp-000 | BASELINE | 89.78% | 41.7 | referência |

Local Blend custa ~33% mais wall-time por step mas entrega +0.77pp — **melhor qualidade/compute em steps**.
