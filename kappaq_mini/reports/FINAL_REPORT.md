# FINAL REPORT — C-MINI-LADDER-v1

## 1. Verdict

**LADDER_SIGNAL**

At CPU tiny-proxy scale (`HuggingFaceTB/SmolLM2-135M`, W3 dense projections, 3 seeds), **rot+mse (A3)** beats both **rot+RTN (A1)** and **mse (A2)** on mean WikiText-2 ppl with a large P2 effect, so the pre-registered LADDER_SIGNAL gate fires. Honesty notes: **P1 failed** (random QR rotation alone worsens ppl vs RTN, with a heavy A1 seed-2 outlier: mean 4511.5 vs median 1269.9); weight-MSE alone (A2) is also worse than RTN (A0); A0/A2 are deterministic so seed std=0. Falsifier=PASS. This is a 135M CPU proxy signal, not a SOTA or 7B claim.

## 2. Hardware & Model

- Hardware: `cpu` (CUDA unavailable; CPU tiny-proxy policy)
- Model: `HuggingFaceTB/SmolLM2-135M`
- Eval: WikiText-2 test slice, 64 sequences × max_length 256 (n_tokens evaluated = 16320)
- Bits: W3 symmetric group_size=128 on dense projections only

## 3. Pre-registration

From `prereg_c_mini_ladder_v1.md`:

- **P1:** mean_ppl(A1) ≤ mean_ppl(A0)
- **P2:** mean_ppl(A3) < mean_ppl(A1)
- **P3:** mean_ppl(A3) < mean_ppl(A2)
- **LADDER_SIGNAL:** P2 and P3 true; P2 effect > max(0.5% relative ppl of A1, 2σ of A3 seed noise)
- **PARTIAL:** P2 true, P3 false
- **NO_SIGNAL:** A3 not better than A0 beyond noise

## 4. Results table

| Arm | Seed0 | Seed1 | Seed2 | Mean | Std |
| --- | --- | --- | --- | --- | --- |
| A0 RTN | 591.992432 | 591.992432 | 591.992432 | 591.992432 | 0.000000 |
| A1 rot+RTN | 1179.901733 | 1269.932983 | 11084.732422 | 4511.522380 | 5692.744865 |
| A2 mse | 917.366394 | 917.366394 | 917.366394 | 917.366394 | 0.000000 |
| A3 rot+mse | 400.064514 | 324.103394 | 381.875488 | 368.681132 | 39.662214 |


### Robustness note (heavy tail)

A1 seed2 ppl=11084.7 dominates the mean. Median±IQR-style read: A1 median=1269.9330; A3 median=381.8755. P2 remains TRUE under medians (381.8755 < 1269.9330).

## 5. Predictions

| ID | Statement | Result | Evidence |
| --- | --- | --- | --- |
| P1 | mean(A1) ≤ mean(A0) | FALSE | 4511.522380 vs 591.992432 |
| P2 | mean(A3) < mean(A1) | TRUE | 368.681132 vs 4511.522380; rel_delta=91.8280%; thresh=1.7583%; effect_ok=True |
| P3 | mean(A3) < mean(A2) | TRUE | 368.681132 vs 917.366394 |

A3 vs A0: mean 368.681132 vs 591.992432; better=True; beyond_noise=True.

## 6. bpw accounting

- Nominal: W3 on dense projections (q/k/v/o, gate/up/down)
- Effective bpw estimate: **5.838181**
- Detail: `{'n_params_total': 134515008.0, 'n_params_dense': 106168320.0, 'n_params_high': 28346688.0, 'n_params_other': 0.0, 'n_groups_dense': 829440.0, 'bits_dense_nominal': 3.0, 'bpw_effective_est': 5.838181327692446, 'fraction_dense': 0.7892674696937906}`
- Kept high-precision (16-bit accounting): embeddings, layer norms, lm_head, biases
- Simulation note: weights remain float tensors; bpw is packing estimate, not kernel-packed storage

## 7. Falsifier outcome

**PASS**

Issues: none

Checks: same n_tokens; rotation flag matches arm; n_modules>0; raw logs present; A1/A3 seeds not collapsed.

## 8. What this does NOT prove

- No out-of-sample selector power for κ_proxy
- No 7B multi-model campaign
- No SOTA W3 claim vs QuIP# / AQLM / QTIP
- CPU 135M proxy may not transfer to 0.5B–7B GPU regimes

## 9. Next actions

Justified next: second small model OR optional A4 sparse holdout (new prereg).

## 10. Ledger pointer

- Path: `kappaq_mini/experiments_log.jsonl`
- Run IDs: ['C-MINI-LADDER-v1-A0-s0', 'C-MINI-LADDER-v1-A0-s1', 'C-MINI-LADDER-v1-A0-s2', 'C-MINI-LADDER-v1-A1-s0', 'C-MINI-LADDER-v1-A1-s1', 'C-MINI-LADDER-v1-A1-s2', 'C-MINI-LADDER-v1-A2-s0', 'C-MINI-LADDER-v1-A2-s1', 'C-MINI-LADDER-v1-A2-s2', 'C-MINI-LADDER-v1-A3-s0', 'C-MINI-LADDER-v1-A3-s1', 'C-MINI-LADDER-v1-A3-s2']
- Row count (all phases): 16
- Generated: 2026-07-30T14:48:30Z
