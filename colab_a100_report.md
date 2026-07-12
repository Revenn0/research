# Relatório — Validação A100 (Google Colab)

**Data:** 2026-07-05 | **GPU:** NVIDIA A100-SXM4-40GB | **Setup:** 1500 steps, batch 512, width×2, bf16, 3 seeds

## Resultado

| Rank | Método | Acc | Δ baseline | Veredito |
|------|--------|-----|------------|----------|
| 1 | multi_scale + LS + wc | **93.37% ± 0.19%** | **+2.32pp** | CONFIRMADA |
| 2 | tri_scale + LS + wc | **93.34% ± 0.22%** | **+2.29pp** | CONFIRMADA |
| 3 | blend + wc | 92.94% ± 0.18% | +1.89pp | CONFIRMADA |
| 4 | blend + LS | 92.56% ± 0.17% | +1.51pp | CONFIRMADA |
| 5 | warmup + cosine | 92.51% ± 0.19% | +1.46pp | CONFIRMADA |
| 6 | k5 + LS | 92.44% ± 0.23% | +1.39pp | CONFIRMADA |
| 7 | local_blend_k5 | 92.31% ± 0.10% | +1.26pp | CONFIRMADA |
| 8 | local_blend | 92.05% ± 0.19% | +1.00pp | CONFIRMADA |
| — | baseline | 91.05% ± 0.31% | — | régua |
| 9 | laplacian_blend | 89.06% ± 1.10% | **−1.99pp** | **REFUTADA** |

Limiar de significância: 0.61pp (2× std do baseline).

## Conclusões

1. **8/9 métodos CONFIRMADOS** — todos os ganhos acima do limiar.
2. **multi_scale e tri_scale no topo (+2.3pp)** — mesma ordem do batch CPU @7200,
   validação cruzada CPU↔GPU consistente.
3. **laplacian_blend REFUTADO** — perde do baseline com instabilidade alta
   (std 1.10%); exp-022/028 rebaixadas para MORTA.
4. Hierarquia confirma a tese: multi-escala NOVEL > stacks compostos > blend puro > baseline.

## Limitações honestas

- Treino curto (13 épocas); teto absoluto não medido — usar `atlas_h100_batch.sh` @24k para isso.
- Apenas FashionMNIST; generalização (CIFAR-10+) não testada.
- Números reportados de run externo no Colab (não reproduzido nesta VM, que não tem GPU).
