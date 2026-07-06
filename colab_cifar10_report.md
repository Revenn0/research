# Relatório — CIFAR-10 Head-to-Head (Colab A100)

**Data:** 2026-07-06 | **GPU:** NVIDIA A100-SXM4-40GB  
**Setup:** STEPS=400, batch=256, width×2, bf16, 3 seeds, TARGET_MINUTES=60

## Ranking

| Rank | Método | Acc | ±std | Δ base | Tipo | Veredito ATLAS |
|------|--------|-----|------|--------|------|----------------|
| 1 | multi_scale_ls_wc | **71.06%** | 0.33% | +37.95pp* | NOVEL | **PROMISSORA** |
| 2 | tri_scale_ls_wc | 71.04% | 0.24% | +37.92pp* | NOVEL | **PROMISSORA** |
| 3 | blend_ls_wc | 70.96% | 0.38% | +37.84pp* | NOVEL | INCONCLUSIVA |
| 4 | se_ls_wc | 69.44% | 0.33% | +36.33pp* | LIT | referência LIT |
| 5 | cbam_ls_wc | 69.14% | 0.58% | +36.02pp* | LIT | — |
| 6 | cbam | 66.13% | 2.71% | +33.01pp* | LIT | — |
| 7 | se_block | 61.02% | 3.36% | +27.90pp* | LIT | — |
| 8 | baseline | 33.12% | **20.30%** | 0 | BASE | **BASELINE INSTÁVEL** |

\*Δ vs baseline **não interpretável** — baseline colapsou (ver abaixo).

## Head-to-head principal (métrica válida)

| | Acc | std |
|---|-----|-----|
| **Melhor NOVEL** (multi_scale_ls_wc) | 71.06% | 0.33% |
| **Melhor LITERATURA** (se_ls_wc) | 69.44% | 0.33% |
| **Δ NOVEL vs LIT** | **+1.62 pp** | |

## Veredito geral do experimento

**PROMISSORA (head-to-head)** — NOVEL supera SE-Net+stack em CIFAR-10 com +1,62 pp e baixa variância.

**INCONCLUSIVA (generalização forte)** — budget de 400 steps é curto; baseline sem LS/wc instável invalida comparação vs régua.

## ⚠️ Problema crítico: baseline 33% ± 20%

- CIFAR-10 aleatório ≈ 10%; 33% com std **20%** indica **pelo menos 1 seed colapsou** (~10% acc) puxando a média.
- Baseline **sem** label_smoothing, warmup ou cosine — enquanto todos os outros têm stack.
- Comparação **não justa** baseline vs métodos com LS+wc.
- Os +37 pp são **artefato** — ignore para narrativa externa.

## O que vale usar

1. **NOVEL vs LITERATURA** no mesmo stack (+1,62 pp) — legítimo
2. Estabilidade: multi/tri_scale std **0,24–0,33%** vs cbam/se_block std **2,7–3,4%**
3. Hierarquia NOVEL ≈ blend > SE+stack > CBAM > SE puro — consistente com FMNIST

## O que NÃO vale usar

- "Ganho de +37 pp no CIFAR"
- "CONFIRMADA" pelo limiar 40,6 pp (derivado do std do baseline quebrado)
- Claim de revolução em escala frontier

## Recomendações

1. **Re-rodar** com `baseline_ls_wc` como régua justa
2. Subir **STEPS** para 1500–2000 mínimo (ou TARGET_MINUTES=90)
3. Manter head-to-head NOVEL vs se_ls_wc / cbam_ls_wc

## Próximo passo ATLAS

- Registrar como **exp-cifar-*** no log
- Promover multi/tri_scale para LM 100M com nota "CIFAR head-to-head +1,62 pp PROMISSORA"
- Não atualizar veredito para REVOLUCIONÁRIA só com este run
