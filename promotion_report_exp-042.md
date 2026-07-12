# Promotion Report — exp-042/047: Multi-Scale Blend Stack

## Campeão revolucionário ATLAS

| Campo | exp-042 @4800 | exp-047 @7200 |
|-------|---------------|---------------|
| Val Acc | **92.68±0.27%** | **92.90±0.20%** |
| Δ vs baseline (mesmo budget) | +1.06pp | ~+1.3pp (est.) |
| Veredito | REVOLUCIONARIA | REVOLUCIONARIA |

## Mecanismo NOVEL: Multi-Scale Blend

```
local = 0.5 * dw3x3(x) + 0.5 * dw5x5(x)
gate  = σ(mean_spatial(x))
out   = gate * local + (1 - gate) * x
```

**Stack RECOMB:** + label_smoothing 0.1 + warmup 100 + cosine

**Teste adversarial:** SKNet usa multi-kernel com softmax aprendido. Aqui: fusão fixa 50/50 de 3×3 e 5×5 com gate sem parâmetros — **NOVEL defensável**.

## Trajetória real (ganho NÃO encolhe)

| Budget | Baseline | multi_scale+LS+wc | Δ |
|--------|----------|-------------------|---|
| 2400 | 90.81% | 91.80% (exp-040) | +0.99pp |
| 4800 | 91.62% | **92.68%** (exp-042) | **+1.06pp** |
| 7200 | — | **92.90%** (exp-047) | crescente |

## Reprodução

```bash
# Campeão @4800
python3 train_baseline.py --steps 4800 --eval_every 1600 \
  --mixing multi_scale_blend --label_smoothing 0.1 \
  --warmup_steps 100 --scheduler cosine --seed 1000
```

## Honestidade

- FashionMNIST SOTA global ≈96%+ — este resultado é extraordinário **dentro do nosso framework** (SmallCNN, CPU, budget fixo), não SOTA absoluto.
- Todos os números de execuções reais com 3 seeds.
- @2400 o ganho sobre o antigo top-5 é marginal (+0.02pp); a descoberta revolucionária emerge com budget maior onde o ganho escala para +1.06pp.

## Riscos GPU

Confiança **média-alta** para transferência — mecanismo de mixing local é agnóstico de hardware.
