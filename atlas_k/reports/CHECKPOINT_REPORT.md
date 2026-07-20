# ATLAS-K CHECKPOINT REPORT

**Status:** Discovery Gate **NOT PASSED**. This is an integrity checkpoint, not a discovery claim.

**Run ID:** `ATLAS-K-2026-07-20`  
**Agent:** https://cursor.com/agents/bc-52a0be72-2f22-4e5c-9045-212929bcbf15  
**Branch:** `cursor/atlas-k-frontier-research-bf15`  
**Hardware:** CPU-only (4 cores, 15GB RAM), PyTorch `2.13.0+cpu`, **NO GPU**  
**Constraint:** L2/L3/L4 scale training is `NOT RUN` (no GPU / insufficient FLOPs budget).

---

## 1. Phase 0 — Literature ingested

| Source | Mechanism extracted | Key unexplored variables |
|--------|---------------------|-------------------------|
| Agents-A1 / arXiv:2606.30616 | Domain-routed OPD + Salient Vocabulary Alignment (truncated reverse KL on teacher top-k); KAG process tuples `(s,a,o,v)` | student-selected support; salient residual of discarded mass; compose SVA geometry with optimizers |
| Gated DeltaNet-2 / arXiv:2605.22791 | Decoupled channel-wise erase `b_t` and write `w_t` + channel decay | echo/residual of erased content; conflict-aware erase; asymmetric state ranks |
| SpinQuant / QuaRot / KronQ | Learned/random rotations + Hessian-aware bits | spectral residual packing; sign-sparse residual codebook on top of GPTQ |
| Muon / NorMuon / AMUSE | Newton–Schulz orthogonalized momentum | salient-subspace Muon (NS only on top-k SVD of momentum) |

Full graph: `atlas_k/kag/knowledge_action_graph.json`

---

## 2. Funnel counts (executed)

| Gate | Count | Notes |
|------|-------|-------|
| L0 hypotheses screened | 10 | 3 promoted, 3 killed (reducible), 4 held |
| L1 tests executed | ≥6 candidate evaluations with multi-seed | all logged in `atlas_k/ledger/ledger.jsonl` |
| L2 / L3 / L4 | **NOT RUN** | hardware |

---

## 3. Leaderboard (honest)

| Rank | Candidate | Domain | Best evidence | Verdict |
|------|-----------|--------|---------------|---------|
| 1 | **H-OPT-SSM** (Salient-Subspace Muon) | training | After LR sweep: AdamW 3.059, Muon 3.096, SS-Muon 3.081 val NLL | `FAIL_L1` (beats Muon slightly, loses to AdamW) |
| 2 | **H-QUANT-HSRP** | quantization | Hadamard+SRP improves KL vs plain SRP but still loses to GPTQ-4 (MSE 0.071 vs 0.061) | `FAIL_L1` |
| 3 | **H-ARCH-KARE** | architecture | L1c (5 seeds): KARE 0.663 vs GDN2 0.765; echo_gap≈0.0003; attn sanity 0.243 | `FAIL_L1` **KILLED** (worse than GDN2; echo useless) |
| 4 | **H-QUANT-SRP / SRP2 / SRC** | quantization | Beat floor-bit baselines; lose to ceil-bit GPTQ/RTN | `FAIL_L1` |
| 5 | **H-ARCH-REM** | architecture | pooled echo ≈ chance | `FAIL_L1` |
| 6 | **H-ARCH-CPE** | architecture | L1e running (PE+shift MQAR + conflict-protected erase) | `IN PROGRESS` |

### Raw numbers that matter (executed, not estimated)

**Quant SRP (5 seeds), eff≈3.25 bits:**
- mean MSE: SRP≈0.071, RTN-3≈0.276, RTN-4≈0.051, GPTQ-4≈0.060
- **Lesson:** spending residual budget on a uniform extra bit beats low-rank spectral residual on this synthetic outlier weight distribution.

**Opt SS-Muon (3 seeds, LR-tuned):**
- AdamW@5e-3: 3.0588 ± ~0.005
- Muon@5e-3: 3.0957 ± ~0.005
- SS-Muon@2e-3: 3.0814 ± ~0.001
- **Lesson:** SS-Muon stabilizes vs poorly-scaled Muon and slightly beats tuned Muon, but does not beat tuned AdamW on this tiny LM proxy.

**Arch REM (3 seeds):** all linear mixers ≈ chance (~3.5%); attention only ~12% — first proxy under-trained.

**Arch KARE L1c (5 seeds, 1500 steps):**
- attn 0.243 ± 0.006 (proxy still invalid without PE/shift)
- gdn2 0.765 ± 0.107
- kare 0.663 ± 0.012
- kare_no_echo 0.663 ± 0.151
- **KARE killed:** −10.1pp vs GDN2; echo_gap ≈ 0

**Quant HSRP (5 seeds):** HSRP MSE 0.071 / KL 0.814 vs GPTQ-4 MSE 0.061 / KL 0.597 — FAIL

---

## 4. Negative-results compost (instructive)

1. **Pooled echo (REM) fails associative recall**; **key-addressable echo (KARE) also fails** — re-injecting erased content interferes with GDN-2 associations (mean −10pp).
2. **Spectral / Hadamard residual packing loses to +1 uniform bit** on outlier-heavy synthetic weights; residual methods must beat GPTQ at *ceil(effective bits)*.
3. **GPTQ-lite with per-row scales on a single column is degenerate (lossless)** — fixed; early FAIL from that bug is logged and superseded.
4. **Muon without Moonlight-style update RMS scaling collapses** at AdamW LRs; fair comparison requires LR sweeps.
5. **MQAR without positional encoding + token-shift underestimates attention** and overestimates same-token delta writes — L1e repairs this.

---

## 5. Exact next actions (for resume)

1. **Finish / verify L1c** (`experiments/l1c_kare_retest.py`): require attention mean ≥ 0.90, then retest KARE vs GDN-2 with 5 seeds × 1500 steps.
2. If KARE passes L1c on MQAR, run **second proxy**: selective copy / induction heads (pre-register before run).
3. Promote **H-ARCH-CONFLICT-ERASE** (modulate erase by key-collision energy) as mutant if KARE fails.
4. Quant path: try **SRC/SRP on GPTQ residual with rotation (QuaRot-style) preprocessing** at matched bits — only if CPU microbench remains discriminative.
5. Acquire **GPU** for any L2 claim (10–50M LM, WikiText/FineWeb sample). Until then, **do not claim Discovery**.
6. Independent **Replicator** + **Falsifier** only after a real multi-proxy L1 pass.

---

## 6. Reproducibility

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python atlas_k/experiments/l1_suite.py --only l0
python atlas_k/experiments/l1_suite.py --only quant
python atlas_k/experiments/l1_suite.py --only opt
python atlas_k/experiments/l1_suite.py --only arch
python atlas_k/experiments/l1b_kare_src.py
python atlas_k/experiments/l1c_kare_retest.py
```

Artifacts: `atlas_k/results/*.json`, append-only `atlas_k/ledger/ledger.jsonl`.

---

## 7. Integrity oath statement

No Discovery is claimed. No fabricated metrics. All reported numbers were produced by executed code and written to the ledger. L2+ marked `NOT RUN`.
