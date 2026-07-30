# Pre-registration: C-MINI-LADDER-v1

**Timestamp (UTC):** 2026-07-30T14:45:00Z  
**Status:** LOCKED before any full arm×seed run  
**Hardware detected:** CPU only (`torch.cuda.is_available() == False`, device=`cpu`)  
**Scope policy applied:** CPU tiny-proxy (≤300M), reduced eval tokens

## Locked model

- **Primary (CPU downgrade):** `HuggingFaceTB/SmolLM2-135M`
- **Why not Qwen2.5-0.5B first:** no GPU / no CUDA; policy requires ≤300M full-model OR layer-subset on ≤1B. Full 135M preferred over layer-subset for clean arm comparison.
- **Dtype load:** float32 on CPU (FP16/BF16 simulated PTQ still applied to dense projections as float tensors)

## Quantization config (locked)

| Field | Value |
| --- | --- |
| Bits | 3 (W3 uniform symmetric grid) |
| Scope | Dense projections only: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (and aliases if present) |
| High precision kept | embeddings, layer norms, lm_head |
| Group size | 128 (remainder: last partial group quantized with its own scale) |
| Symmetric | yes (max-abs scale; zero-point = 0) |
| Rotation family | QR-orthogonal (fixed per seed); same R reused for A1 and A3 within a seed |
| MSE clip | grid-search scale multiplier on max-abs; minimize weight MSE; calib-free (weight-only) |
| Seeds | 0, 1, 2 |

## Eval (CPU pressure)

| Field | Value |
| --- | --- |
| Corpus | WikiText-2 test (HF `wikitext` / `wikitext-2-raw-v1`); local fallback if download fails |
| Sequences | first **64** contiguous chunks |
| Max length | **256** |
| Metric | perplexity (exp mean NLL over tokens) |
| Arms | A0 RTN, A1 rot+RTN, A2 mse, A3 rot+mse |

## Predictions (P1–P3)

- **P1:** `mean_ppl(A1) ≤ mean_ppl(A0)` — rotation does not destroy; preferably helps
- **P2:** `mean_ppl(A3) < mean_ppl(A1)` — mse unlocks after rotation
- **P3:** `mean_ppl(A3) < mean_ppl(A2)` — rotation matters for mse benefit (sequential, not pure additivity)

## Pass / kill thresholds

- **LADDER_SIGNAL:** P2 and P3 both true; effect size on P2 > max(0.5% relative ppl of A1, 2σ of seed noise on A3)
- **PARTIAL:** P2 true, P3 false
- **NO_SIGNAL:** A3 not better than A0 beyond noise (not global death of KappaQ — scale/limit note)
- **INVALID:** falsifier failure (token mismatch, FP16 leftover bug, rotation missing where claimed, seed collapse)

## Fair arms

Same model, same bit budget definition, same eval slice, same seeds, same group size / packing for all arms. Rotation seed policy: for seed `k`, draw orthogonal R with `torch.Generator` seeded by `k` once; reuse for A1 and A3.

## bpw reporting

- Nominal: W3 on dense projections
- Effective bpw: estimate over full parameter count with embeddings/norms/lm_head at 16-bit equivalent storage assumption for accounting (weights remain float in sim)
