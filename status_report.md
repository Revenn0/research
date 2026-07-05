# ATLAS — Status Report

**Última atualização:** 2026-07-05 12:20 UTC  
**Hardware:** CPU Linux, PyTorch 2.12.1+cpu, ~41 steps/s (SmallCNN, FashionMNIST, batch=128)  
**Experimento ativo:** ciclo inicial concluído; fila: combinações de schedule + regularização

## Resumo executivo

Infraestrutura ATLAS operacional com 14 experimentos reais executados. Baseline bem tunado (89.78% val acc @1200 steps, 3 seeds). **Campeão atual: warmup 100 steps + cosine LR decay** (`exp-007`/`exp-011`), com ganho consistente que **aumenta** em budget 2× (+0.57 pp absolutos vs baseline @2400 steps).

Nenhuma técnica genuinamente nova à literatura emergiu neste ciclo — o campeão é redescoberta de prática estabelecida (warmup + cosine). Várias hipóteses foram refutadas com lições úteis.

## Hardware medido

| Métrica | Valor |
|---------|-------|
| Throughput médio | 41.7 steps/s |
| Tempo/run @1200 steps | ~29 s |
| Tempo/run @2400 steps | ~58 s |
| Dataset | FashionMNIST (60k train / 10k val) |
| Modelo | SmallCNN (~200k params) |

## Resultados-chave (budget 1200 steps, 3 seeds)

| ID | Variante | Val Acc (mean±std) | Δ vs baseline | Veredito |
|----|----------|-------------------|---------------|----------|
| exp-000 | baseline | 89.78±0.31% | — | BASELINE |
| exp-007 | warmup100+cosine | **90.07±0.32%** | **+0.29pp** | PROMISSORA |
| exp-002 | label_smoothing 0.1 | 89.98±0.18% | +0.20pp | INCONCLUSIVA |
| exp-003 | EMA 0.999 | 88.25±0.07% | -1.53pp | MORTA |
| exp-005 | BatchNorm | 89.58±0.67% | -0.20pp | MORTA |

## Validação de escala (2400 steps, regra 3)

| ID | Variante | Val Acc | Δ vs baseline @2400 |
|----|----------|---------|---------------------|
| exp-010 | baseline | 90.81±0.31% | — |
| exp-011 | warmup+cosine | **91.38±0.14%** | **+0.57pp** |
| exp-013 | label_smoothing | 90.89±0.07% | +0.08pp |

O ganho do campeão **não encolheu** — cresceu proporcionalmente. Veredito PROMISSORA mantido.

## Lições aprendidas

1. **EMA decay=0.999 é inadequado** para treinos curtos (<5k steps): avaliação intermediária mostra acc ~75% @400 steps.
2. **BatchNorm degrada throughput em CPU** (~29 vs ~42 steps/s) sem ganho de acurácia.
3. **Combos nem sempre somam**: LS + warmup + cosine (exp-008) ≤ warmup+cosine isolado.
4. **Cutout** prejudica neste modelo pequeno — provavelmente precisa de arquitetura maior.

## Próximos experimentos (fila)

- [ ] `warmup+cosine` + `label_smoothing` @2400 (combo no campeão)
- [ ] Warmup proporcional (8% dos steps) em budgets variados
- [ ] Patch: cosine com restart único em 50% do budget
- [ ] Grad clip 1.0 + warmup+cosine (regularização de gradiente)

## Arquivos de estado

- `experiments_log.jsonl` — 14 entradas
- `leaderboard.md` — ranking atualizado
- `promotion_report_exp-011.md` — campeão para validação GPU
