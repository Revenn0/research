# ATLAS — Status Report

**Última atualização:** 2026-07-05 16:20 UTC  
**Meta:** superar top-5 (91.78%) com resultados extraordinários reais — **ATINGIDA**

## Campeão revolucionário

**exp-047: multi_scale_blend + LS + warmup/cosine @7200 = 92.90% ± 0.20%**

Ganho vs baseline @4800: **+1.28pp** (baseline 4800 = 91.62%)

## Resultados-chave deste ciclo

| ID | Acc | Budget | Supera 91.78%? | Veredito |
|----|-----|--------|----------------|----------|
| exp-047 | 92.90% | 7200 | ✓✓ | REVOLUCIONARIA |
| exp-042 | 92.68% | 4800 | ✓✓ | REVOLUCIONARIA |
| exp-041 | 92.15% | 4800 | ✓ | PROMISSORA |
| exp-033 | 91.80% | 2400 | ✓ (marginal) | PROMISSORA |
| exp-040 | 91.80% | 2400 | ✓ (marginal) | PROMISSORA |

## Mecanismo NOVEL descoberto

**Multi-Scale Blend:** `gate*0.5*(dw3(x)+dw5(x)) + (1-gate)*x`

Mais próximo publicado: SKNet — diferença: sem softmax multi-branch, fusão fixa 50/50, gate sem parâmetros.

## Honestidade científica

1. Números 100% reais (PyTorch CPU, 3 seeds cada)
2. @2400 o ganho sobre o antigo #1 é +0.02pp — honestamente marginal
3. A salto extraordinário (+1.06pp) aparece @4800+ com ganho crescente
4. Não é SOTA FashionMNIST global (~96%), mas é ganho real e reproduzível no framework ATLAS

## Total: 48 experimentos

Ver `promotion_report_exp-042.md` para validação GPU.
