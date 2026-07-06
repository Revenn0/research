# ATLAS — Critérios de Veredito @100M

**Escopo:** batch `exp-100m-001` … `exp-100m-012` + `exp-100m-baseline`  
**Escala:** ~107,5M parâmetros · FashionMNIST · 1000 steps × 3 seeds  
**Métrica principal:** `Δ = best_val_acc_mean − baseline_100M` (em pontos percentuais, pp)

---

## Tabela de vereditos

| Veredito | Δ vs baseline @100M | Condição extra | O que você pode dizer |
|----------|---------------------|----------------|---------------------|
| **REFUTADA** | **≤ −0,50 pp** | — | Mecanismo **piora** em escala 100M. Descartar nesta linha. |
| **MORTA** | ≤ −1,00 pp | ou colapso (acc < 70%, NaN, instável) | Falha clara. Não promover. |
| **INCONCLUSIVA** | **−0,49 a +0,29 pp** | ou `std > 0,40 pp` com Δ pequeno | Ruído / empate técnico. Precisa mais budget ou outro dataset. |
| **PROMISSORA** | **+0,30 a +0,79 pp** | `std ≤ 0,35 pp` nas 3 seeds | Ganho **real mas modesto** em 100M. Vale próxima fase (CIFAR / LM). |
| **REVOLUCIONÁRIA** | **≥ +0,80 pp** | top-3 do ranking 100M **e** `std ≤ 0,30 pp` | Ganho **forte e estável** em 100M no mesmo benchmark. |
| **BASELINE** | referência (0 pp) | `exp-100m-baseline` | Régua — não ranqueia contra si mesma. |

---

## Regras de desempate

1. Maior `best_val_acc_mean` vence.
2. Empate (< 0,10 pp): menor `best_val_acc_std` vence.
3. NOVEL ganha sobre RECOMB se Δ dentro de 0,15 pp (mesma eficácia, mais original).

---

## O que cada veredito autoriza (e o que NÃO autoriza)

| Veredito @100M | Pode dizer | Não pode dizer |
|----------------|------------|----------------|
| PROMISSORA | *"Mantém vantagem em ~100M params no FMNIST"* | *"Revolucionário em 100B"* |
| REVOLUCIONÁRIA | *"Ganho robusto persiste em 100M — candidato a escala LM"* | *"Supera frontier models"* |
| REFUTADA | *"Não escala — piora em 100M"* | — |
| INCONCLUSIVA | *"Precisa CIFAR-10 ou LM para decidir"* | Qualquer claim forte |

---

## Ponte para 100B (escada de evidência)

```text
100M FMNIST (este batch)  →  sobrevive em capacidade?
        ↓
CIFAR-10 vs SE/CBAM       →  generaliza além de FMNIST?
        ↓
LM 100M TokenBlend        →  funciona em arquitetura tipo LM?
        ↓
1B → 7B                   →  lei de escala?
        ↓
100B frontier             →  só com benchmark público (MMLU, coding…)
```

**Regra de ouro:** veredito @100M é **necessário**, nunca **suficiente**, para claim em 100B.

---

## Aplicação automática (quando baseline existir)

```python
def verdict_100m(delta_pp: float, std_pp: float) -> str:
    if delta_pp <= -1.0:
        return "MORTA"
    if delta_pp <= -0.5:
        return "REFUTADA"
    if delta_pp < 0.3 or std_pp > 0.4:
        return "INCONCLUSIVA"
    if delta_pp < 0.8:
        return "PROMISSORA"
    if std_pp <= 0.3:
        return "REVOLUCIONARIA"
    return "INCONCLUSIVA"  # ganho alto mas instável
```

---

## Referência histórica (escala pequena → Colab A100)

| Escala | Baseline | Campeão | Δ | Veredito |
|--------|----------|---------|---|----------|
| CPU @7200 | ~91,62% | multi_scale+LS+wc 92,90% | +1,28 pp | REVOLUCIONÁRIA |
| Colab A100 | 91,05% | multi_scale 93,37% | +2,32 pp | CONFIRMADA |
| Colab laplacian | 91,05% | 89,06% | −1,99 pp | REFUTADA |

**Expectativa @100M:** Δ tende a **encolher** (dataset satura com 107M params). Por isso os limiares aqui são **menores** que em escala pequena (+0,8 pp já é REVOLUCIONÁRIA em 100M).

---

## Status parcial (atualiza com o batch)

| ID | Val Acc | Δ base | Veredito |
|----|---------|--------|----------|
| exp-100m-001 | 89,78% ± 0,26% | *aguarda baseline* | INCONCLUSIVA |
| exp-100m-002..012 | — | — | rodando |
| exp-100m-baseline | — | 0 pp | pendente |

*Atualizado automaticamente em `promissora_100m_tracker.md` quando o batch fechar.*
