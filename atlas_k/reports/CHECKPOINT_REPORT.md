# ATLAS-K CHECKPOINT REPORT

**Status:** Discovery Gate **NOT PASSED**. Integrity checkpoint only — no discovery claimed.

| Field | Value |
|-------|-------|
| Run ID | `ATLAS-K-2026-07-20` |
| Agent | https://cursor.com/agents/bc-52a0be72-2f22-4e5c-9045-212929bcbf15 |
| Branch | `cursor/atlas-k-frontier-research-bf15` |
| PR | https://github.com/Revenn0/research/pull/7 |
| Hardware | CPU-only, 4 cores, PyTorch 2.13.0+cpu, **NO GPU** |
| L2/L3/L4 | **NOT RUN** |

---

## Phase 0 — Knowledge-Action Graph

Ingested: Agents-A1 (arXiv:2606.30616), Gated DeltaNet-2 (2605.22791), SpinQuant/QuaRot/KronQ, Muon/NorMuon/AMUSE.

Unexplored-variable fuel → hypotheses in `atlas_k/kag/knowledge_action_graph.json` and `hypotheses/PREREGISTER.md`.

---

## Funnel counts

| Gate | N | Result |
|------|---|--------|
| L0 screened | 10 | 3 promote, 3 kill, 4 hold |
| L1 executed | 8 candidate evaluations (multi-seed) | **0 PASS_L1** |
| L2+ | 0 | NOT RUN |

---

## Leaderboard (all executed)

| Candidate | Mean evidence | Verdict |
|-----------|---------------|---------|
| H-ARCH-CPE (conflict-protected erase) | MQAR L1e: CPE 0.819 vs GDN2 0.817 vs Attn 0.816 (5 seeds) | **FAIL_L1** (Δ=+0.002 ≪ 0.05) |
| H-OPT-SSM (salient-subspace Muon) | NLL: AdamW 3.059, SS-Muon 3.081, Muon 3.096 (LR-swept) | **FAIL_L1** |
| H-QUANT-HSRP | MSE 0.071 vs GPTQ-4 0.061 | **FAIL_L1** |
| H-ARCH-KARE | MQAR L1c: 0.663 vs GDN2 0.765; echo_gap≈0 | **FAIL_L1 / KILLED** |
| H-QUANT-SRP/SRP2/SRC | beat floor bits; lose to ceil GPTQ/RTN | **FAIL_L1** |
| H-ARCH-REM | ≈ chance | **FAIL_L1** |

---

## Instructive negatives (compost)

1. Re-injecting erased delta-rule content (REM/KARE) **hurts** MQAR (−10pp vs GDN2 in L1c).
2. Spectral/sign residuals beat n-bit GPTQ but **lose to (n+1)-bit GPTQ** at matched effective bits.
3. Muon needs RMS scaling + LR sweep; SS-Muon ≈ AdamW but does not beat it here.
4. MQAR without PE + causal/shift invalidates attention baselines (fixed in L1e → all ~82%).

---

## Exact next actions

1. GPU environment for L2 (10–50M LM) — mandatory before any discovery claim.
2. Mutate CPE: protect based on *key-collision energy* `||Sᵀk||` not value-read cos (pre-register first).
3. Second proxy only after a method beats GDN2 by ≥5pp on fixed MQAR with attn≥0.90 (more steps/PE).
4. Quant: try residual packing **only on Hessian-top columns** (salient-column SRC) at matched bits.
5. Do not declare success without L1×2 proxies + L4 replication.

---

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python atlas_k/experiments/l1_suite.py --only l0
python atlas_k/experiments/l1_suite.py --only quant
python atlas_k/experiments/l1_suite.py --only opt
python atlas_k/experiments/l1_suite.py --only arch
python atlas_k/experiments/l1b_kare_src.py
python atlas_k/experiments/l1c_kare_retest.py
python atlas_k/experiments/l1d_hsrp.py
python atlas_k/experiments/l1e_cpe.py
```

Ledger: `atlas_k/ledger/ledger.jsonl` (append-only). Results: `atlas_k/results/*.json`.

**Integrity:** Every number above was produced by executed code. No fabrication. Discovery Gate closed without a pass.
