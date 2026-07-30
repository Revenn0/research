# KappaQ Mini-Test Swarm (C-MINI-LADDER-v1)

Focused W3 mechanism ladder probe: **RTN → rot → mse → rotmse** on a small model.

## Hardware lock (this run)

- Detected: **CPU only**
- Model: `HuggingFaceTB/SmolLM2-135M` (CPU tiny-proxy ≤300M)
- Eval: 64 × 256 WikiText-2 (smoke: 2 × 128)

## Quick start

```bash
pip install -r requirements.txt
python tests/test_quant_unit.py
python src/run_arm.py --arm A0 --seed 0 --smoke
python src/run_arm.py --arm A0 --seed 0   # full matrix: A0-A3 × seeds 0,1,2
```

## Layout

See `prereg_c_mini_ladder_v1.md`, `experiments_log.jsonl`, `reports/FINAL_REPORT.md`.

## Integrity

No fabricated metrics. Append-only ledger. Negative results stay.
