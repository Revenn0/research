# Promotion Report — exp-027: Local Blend + Label Smoothing @2400

## Campeão geral ATLAS

| Campo | Valor |
|-------|-------|
| ID | exp-027 |
| Mecanismo | Local Blend (NOVEL) + Label Smoothing 0.1 (CONHECIDA) |
| Rótulo | **RECOMB** |
| Veredito | **PROMISSORA** (#10 da meta) |

## Ganho vs baseline

| Budget | Baseline | blend+LS | Δ |
|--------|----------|----------|---|
| 1200 | 89.78% | 90.67% (exp-025) | +0.89pp |
| 2400 | 90.81% | **91.78%** | **+0.97pp** |

Ganho **cresce** com budget 2× (regra 3 satisfeita).

## Comando reprodução

```bash
python3 train_baseline.py --steps 2400 --eval_every 800 \
  --mixing local_blend --label_smoothing 0.1 --seed 1000
```

## Expectativa GPU

Confiança **média-alta** — combina o melhor mixing NOVEL com regularização estabelecida.

## Script Colab

```python
!pip install -q torch torchvision
import subprocess, json
def run(s):
    out = subprocess.check_output([
        "python","train_baseline.py","--steps","2400","--eval_every","800",
        "--mixing","local_blend","--label_smoothing","0.1","--seed",str(s)
    ], text=True)
    return json.loads(out)
accs=[run(s)["best_val_acc"] for s in [1000,1001,1002]]
print(f"blend+LS: {sum(accs)/3:.4f}")
```
