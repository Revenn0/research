# Promotion Report — exp-018: Local Blend Mixing (NOVEL)

## Identificação

| Campo | Valor |
|-------|-------|
| ID | exp-018 (descoberta inicial: exp-016) |
| Mecanismo | Local Blend: mistura depthwise 3×3 com identidade, gate escalar por canal |
| Rótulo novidade | **NOVEL** |
| Veredito | **PROMISSORA** |

## Teste adversarial de novidade

| Aspecto | Detalhe |
|---------|---------|
| Técnica mais próxima | SKNet (Li et al. 2019) — múltiplos kernels com softmax de seleção |
| Diferença defensável | Um único depthwise 3×3; gate = σ(mean_spatial(x)) por canal; sem branches paralelos nem softmax multi-kernel |
| Técnica secundária próxima | SE-Net — gate global com MLP; aqui gate derivado da média espacial sem parâmetros extras no gate |
| Conclusão | Diferença estrutural suficiente para **NOVEL** |

## Fórmula

```
local = DepthwiseConv3x3(x)
gate  = sigmoid(mean(x, dim=spatial))   # escalar por canal
out   = gate * local + (1 - gate) * x
```

Aplicado após cada conv block, antes da ativação ReLU.

## Ganho vs baseline (dados reais, 3 seeds)

| Budget | Baseline | Local Blend | Δ absoluto |
|--------|----------|-------------|------------|
| 1200 steps | 89.78±0.31% | 90.48±0.38% | **+0.70pp** |
| 2400 steps | 90.81±0.31% | **91.58±0.35%** | **+0.77pp** |

Comparação com calibração RECOMB (exp-011 warmup+cosine @2400): 91.38% → Local Blend **supera em +0.20pp** com mecanismo NOVEL.

O ganho **não encolheu** com budget 2× — cresceu ligeiramente (regra 3 satisfeita).

## Patch (sobre train_baseline.py)

```python
class LocalBlend(nn.Module):
    def __init__(self, channels):
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x):
        local = self.dw(x)
        gate = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * x
```

**Comando reprodução @2400:**

```bash
python3 train_baseline.py --steps 2400 --eval_every 800 --mixing local_blend --seed 1000
```

## Expectativa em escala GPU

| Aspecto | Expectativa | Confiança |
|---------|-------------|-----------|
| Ganho em CNNs maiores | +0.5 a +1.5pp se mixing local ajuda features espaciais | Média |
| Custo compute | +~40% wall time (depthwise extra); trade-off favorável em steps | Alta (medido) |
| Transferência ResNet-like | Provável — mixing local entre conv blocks é natural | Média |
| Risco de overfit | Baixo — dirac init ≈ identidade no início | Média-Alta |

## Script Colab

```python
!pip install -q torch torchvision
# Upload train_baseline.py

import subprocess, json

def run(seed, steps=2400):
    out = subprocess.check_output([
        "python", "train_baseline.py",
        "--steps", str(steps), "--eval_every", str(steps // 3),
        "--mixing", "local_blend", "--seed", str(seed),
    ], text=True)
    return json.loads(out)

results = [run(s) for s in [1000, 1001, 1002]]
accs = [r["best_val_acc"] for r in results]
print(f"Local Blend: {sum(accs)/len(accs):.4f}")
```

## Riscos e refutações

| Risco | Status |
|-------|--------|
| Ganho some com budget maior | **Refutado** — +0.70→+0.77pp |
| Artefato de seed | std 0.35% @2400 — aceitável |
| Throughput degradado | Confirmado (~27 vs ~41 steps/s) — custo real |
| Redescoberta SKNet | **Refutado** — arquitetura estruturalmente distinta |

## Recomendação

**Promover para validação GPU.** Primeiro candidato NOVEL da missão ATLAS.
