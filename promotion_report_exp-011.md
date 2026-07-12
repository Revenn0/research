# Promotion Report — exp-011: Warmup + Cosine LR Decay

## Identificação

| Campo | Valor |
|-------|-------|
| ID | exp-011 (descoberta inicial: exp-007) |
| Mecanismo | Linear warmup (100 steps) + cosine annealing do learning rate |
| Novidade | **REDESCOBERTA** — prática estabelecida (Goyal et al. 2017; Transformers/CV) |
| Veredito | **PROMISSORA** |

## Ganho vs baseline (dados reais)

| Budget | Baseline | Warmup+Cosine | Δ absoluto | Δ relativo |
|--------|----------|---------------|------------|------------|
| 1200 steps | 89.78±0.31% | 90.07±0.32% | +0.29pp | +0.32% |
| 2400 steps | 90.81±0.31% | 91.38±0.14% | +0.57pp | +0.63% |

Incerteza: std entre 3 seeds (1000, 1001, 1002). Ganho **aumentou** com budget 2× — não é artefato de convergência precoce.

## Patch (mínimo, sobre train_baseline.py)

Já integrado via flags CLI — sem reescrita do trainer:

```python
# Em lr_at_step():
if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
    return cfg.lr * (step + 1) / cfg.warmup_steps
if cfg.scheduler == "cosine":
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
```

**Comando de reprodução (CPU):**

```bash
python3 train_baseline.py --steps 2400 --eval_every 800 --warmup_steps 100 --scheduler cosine --seed 1000
```

**Comando multi-seed:**

```bash
python3 atlas_run.py --id exp-011 --name warmup_cosine_2400steps --verdict PROMISSORA --seeds 3 -- \
  --steps 2400 --eval_every 800 --warmup_steps 100 --scheduler cosine
```

## Expectativa em escala GPU

| Aspecto | Expectativa | Confiança |
|---------|-------------|-----------|
| Ganho em CNNs maiores (ResNet-18+) | +0.3 a +1.0pp val acc | **Média** — warmup+cosine é padrão em muitos benchmarks |
| Ganho em Transformers | Já incorporado na maioria dos trainers | **Alta** — pode ser neutro se baseline já usa |
| Transferência para outros datasets | Provável benefício em treinos curtos/médios | **Média** |
| Risco de overfitting reduzido | Cosine annealing suaviza final do treino | **Média-Alta** |

**Hipótese:** Em GPU com treinos longos onde o baseline já usa cosine, o ganho pode desaparecer. Validar contra seu trainer GPU existente.

## Script Colab (pronto)

```python
# Colab: ATLAS exp-011 validation
!pip install -q torch torchvision

import subprocess, json

def run(seed, steps=2400):
    out = subprocess.check_output([
        "python", "train_baseline.py",
        "--steps", str(steps),
        "--eval_every", str(steps // 3),
        "--warmup_steps", "100",
        "--scheduler", "cosine",
        "--seed", str(seed),
    ], text=True)
    return json.loads(out)

# Upload train_baseline.py ao Colab antes de executar
baseline = [run(s, steps=2400) for s in [1000, 1001, 1002]]
accs = [r["best_val_acc"] for r in baseline]
print(f"Warmup+Cosine: {sum(accs)/len(accs):.4f} ± {max(accs)-min(accs):.4f} (range)")
```

## Riscos e refutações tentadas

| Risco | Status |
|-------|--------|
| Ganho some com budget maior | **Refutado** — ganho dobrou de 0.29→0.57pp |
| Alta variância entre seeds | **Parcial** — std 0.14% @2400, aceitável |
| Custo compute extra | **Nenhum** — mesmo número de steps, mesma wall time |
| Combo com LS piora | **Confirmado** — exp-008 não supera exp-007 |
| Técnica já no seu pipeline | Verificar — se sim, MORTA no seu contexto |

## Recomendação

Promover para validação GPU com:
1. Mesmo budget de steps que seu baseline GPU
2. Warmup = min(100, 8% do total de steps)
3. Comparar contra baseline que **não** já usa cosine (muitos trainers modernos já usam)

Se confirmado em GPU, integrar como default no trainer de referência.
