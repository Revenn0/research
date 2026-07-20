# ATLAS-K Hypotheses — Pre-registration

## Promoted to L1

### H-ARCH-REM — Residual Echo Mixing
- **Formula**: After delta-rule erase of `erase = b ⊙ (S k)`, write echo `E ← γ⊙E + (1-γ)⊙P(erase)`; output `y = [S k ; β E]`.
- **Nearest neighbors**: Gated DeltaNet, KDA, Gated DeltaNet-2 (erase is irreversible).
- **Delta**: Second-state recovery path for erased associations.
- **Prediction**: MQAR acc mean(REM) − mean(GatedDelta) ≥ 0.05 and > 2σ; echo ablation gap ≥ 0.03.
- **Kill**: Fails margin OR echo ablation matches REM.

### H-QUANT-SRP — Spectral Residual Packing
- **Formula**: `Ŵ = Q_n(W) + Q_m(U_r) diag(Σ_r) Q_m(V_r)` with rank-r SVD of `W−Q_n(W)`.
- **Nearest neighbors**: GPTQ residual redistribution, QuIP lattice codes, mixed-precision bit allocation.
- **Delta**: Explicit quantized spectral residual at matched average bit budget vs spending bits on finer uniform grid.
- **Prediction**: MSE ≤ 0.9 × RTN_matched and KL < RTN_matched; also beats or matches GPTQ-lite on MSE or KL.
- **Kill**: Fails MSE or KL vs matched-bit baselines.

### H-OPT-SSM — Salient-Subspace Muon
- **Formula**: `M = M_sal + M_res` via top-k SVD; `Δ = NS(M_sal)·scale + Adam(M_res)`.
- **Nearest neighbors**: Muon (full NS), SOAP (eigenbasis Adam).
- **Delta**: Orthogonalize only salient momentum subspace; Adam on complement.
- **Prediction**: val NLL improves ≥0.02 vs AdamW and ≥0.01 vs full Muon.
- **Kill**: Does not beat both.

## Held / Killed
See `results/l0_screen.json` and ledger.
