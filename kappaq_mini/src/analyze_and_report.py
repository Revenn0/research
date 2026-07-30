#!/usr/bin/env python3
"""Falsifier + analysis for C-MINI-LADDER-v1."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments_log.jsonl"
REPORT = ROOT / "reports" / "FINAL_REPORT.md"
LEADERBOARD = ROOT / "leaderboard.md"
STATUS = ROOT / "status_report.md"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_runs() -> List[Dict[str, Any]]:
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return [r for r in rows if r.get("phase") == "run" and not str(r.get("id", "")).startswith("SMOKE")]


def mean_std(xs: List[float]):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def append_ledger(row: Dict[str, Any]) -> None:
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    runs = load_runs()
    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in runs:
        by_arm[r["arm"]][int(r["seed"])] = r

    arms = ["A0", "A1", "A2", "A3"]
    table = {}
    issues = []

    # Falsifier checks
    token_counts = []
    for arm in arms:
        for seed in (0, 1, 2):
            r = by_arm.get(arm, {}).get(seed)
            if r is None:
                issues.append(f"MISSING {arm} seed{seed}")
                continue
            m = r.get("metrics") or {}
            ppl = m.get("ppl")
            ntok = m.get("n_tokens")
            if ppl is None or not (ppl == ppl) or ppl <= 0:
                issues.append(f"BAD_PPL {arm} s{seed}: {ppl}")
            token_counts.append(ntok)
            info = r.get("arm_info_summary") or {}
            want_rot = arm in ("A1", "A3")
            if bool(info.get("rotation_applied")) != want_rot:
                issues.append(f"ROT_MISMATCH {arm} s{seed}: {info}")
            if info.get("n_modules", 0) <= 0:
                issues.append(f"NO_MODULES {arm} s{seed}")
            # raw log must exist
            raw = ROOT / (r.get("raw_log_path") or "")
            if not raw.exists():
                issues.append(f"MISSING_RAW {arm} s{seed}: {raw}")

    if token_counts and len(set(token_counts)) != 1:
        issues.append(f"TOKEN_MISMATCH counts={token_counts}")

    # Seed differentiation for stochastic rotation: A1/A3 ppl should not be identical across seeds
    for arm in ("A1", "A3"):
        ppls = []
        for seed in (0, 1, 2):
            r = by_arm.get(arm, {}).get(seed)
            if r:
                ppls.append(r["metrics"]["ppl"])
        if len(ppls) == 3 and len(set(round(x, 8) for x in ppls)) == 1:
            issues.append(f"SEED_COLLAPSE {arm}: identical ppl {ppls}")

    falsifier = "PASS" if not issues else "FAIL"

    for arm in arms:
        ppls = []
        for seed in (0, 1, 2):
            r = by_arm.get(arm, {}).get(seed)
            ppls.append(None if r is None else r["metrics"]["ppl"])
        finite = [x for x in ppls if x is not None]
        mu, sd = mean_std(finite) if finite else (float("nan"), float("nan"))
        table[arm] = {"ppls": ppls, "mean": mu, "std": sd}

    names = {"A0": "RTN", "A1": "rot+RTN", "A2": "mse", "A3": "rot+mse"}
    m0, m1, m2, m3 = (table[a]["mean"] for a in arms)

    p1 = m1 <= m0
    p2 = m3 < m1
    p3 = m3 < m2

    # effect size on P2
    rel = abs(m1 - m3) / m1 if m1 and m1 == m1 else float("nan")
    two_sigma = 2 * table["A3"]["std"] if table["A3"]["std"] == table["A3"]["std"] else float("nan")
    thresh = max(0.005, (two_sigma / m1) if m1 else 0.005)
    p2_effect_ok = (rel > thresh) if (rel == rel and thresh == thresh) else False

    if falsifier != "PASS":
        verdict = "INVALID"
    elif p2 and p3 and p2_effect_ok:
        verdict = "LADDER_SIGNAL"
    elif p2 and not p3:
        verdict = "PARTIAL"
    else:
        verdict = "NO_SIGNAL"

    # A3 vs A0 beyond noise
    a3_better = m3 < m0
    a3_vs_a0_noise = abs(m0 - m3) > max(0.005 * m0, 2 * table["A3"]["std"]) if m0 == m0 else False

    bpw = None
    for arm in arms:
        for seed in (0, 1, 2):
            r = by_arm.get(arm, {}).get(seed)
            if r and r.get("bpw_effective_est") is not None:
                bpw = r["bpw_effective_est"]
                bpw_detail = r.get("bpw_detail")
                break
        if bpw is not None:
            break

    model = runs[0]["model"] if runs else "UNKNOWN"
    hardware = runs[0]["hardware"] if runs else "UNKNOWN"
    n_tokens = token_counts[0] if token_counts else "NA"

    # leaderboard
    lines = [
        "# Leaderboard — C-MINI-LADDER-v1",
        "",
        f"Updated: {utc()}",
        "",
        "| Arm | Seed0 | Seed1 | Seed2 | Mean | Std |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        p0, p1s, p2s = table[arm]["ppls"]
        def fmt(x):
            return "—" if x is None else f"{x:.4f}"
        lines.append(
            f"| {arm} {names[arm]} | {fmt(p0)} | {fmt(p1s)} | {fmt(p2s)} | {fmt(table[arm]['mean'])} | {fmt(table[arm]['std'])} |"
        )
    LEADERBOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # FINAL REPORT
    next_actions = {
        "LADDER_SIGNAL": "Justified next: second small model OR optional A4 sparse holdout (new prereg).",
        "PARTIAL": "mse helps after rotation but rotation does not beat mse alone; consider larger scale / different rotation family (new prereg).",
        "NO_SIGNAL": "Stop or change scale/family; new prereg required. This is not global death of KappaQ.",
        "INVALID": "Fix methodology (see falsifier) and rerun.",
    }[verdict]

    report = f"""# FINAL REPORT — C-MINI-LADDER-v1

## 1. Verdict

**{verdict}**

At CPU tiny-proxy scale (`{model}`, W3 dense projections, 3 seeds), the sequential unlock claims P1–P3 evaluate to P1={p1}, P2={p2}, P3={p3} with P2 relative effect={rel:.4%} vs threshold={thresh:.4%}. Falsifier={falsifier}.

## 2. Hardware & Model

- Hardware: `{hardware}` (CUDA unavailable; CPU tiny-proxy policy)
- Model: `{model}`
- Eval: WikiText-2 test slice, 64 sequences × max_length 256 (n_tokens evaluated = {n_tokens})
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
| A0 RTN | {table['A0']['ppls'][0]:.6f} | {table['A0']['ppls'][1]:.6f} | {table['A0']['ppls'][2]:.6f} | {table['A0']['mean']:.6f} | {table['A0']['std']:.6f} |
| A1 rot+RTN | {table['A1']['ppls'][0]:.6f} | {table['A1']['ppls'][1]:.6f} | {table['A1']['ppls'][2]:.6f} | {table['A1']['mean']:.6f} | {table['A1']['std']:.6f} |
| A2 mse | {table['A2']['ppls'][0]:.6f} | {table['A2']['ppls'][1]:.6f} | {table['A2']['ppls'][2]:.6f} | {table['A2']['mean']:.6f} | {table['A2']['std']:.6f} |
| A3 rot+mse | {table['A3']['ppls'][0]:.6f} | {table['A3']['ppls'][1]:.6f} | {table['A3']['ppls'][2]:.6f} | {table['A3']['mean']:.6f} | {table['A3']['std']:.6f} |

## 5. Predictions

| ID | Statement | Result | Evidence |
| --- | --- | --- | --- |
| P1 | mean(A1) ≤ mean(A0) | {"TRUE" if p1 else "FALSE"} | {m1:.6f} vs {m0:.6f} |
| P2 | mean(A3) < mean(A1) | {"TRUE" if p2 else "FALSE"} | {m3:.6f} vs {m1:.6f}; rel_delta={rel:.4%}; thresh={thresh:.4%}; effect_ok={p2_effect_ok} |
| P3 | mean(A3) < mean(A2) | {"TRUE" if p3 else "FALSE"} | {m3:.6f} vs {m2:.6f} |

A3 vs A0: mean {m3:.6f} vs {m0:.6f}; better={a3_better}; beyond_noise={a3_vs_a0_noise}.

## 6. bpw accounting

- Nominal: W3 on dense projections (q/k/v/o, gate/up/down)
- Effective bpw estimate: **{bpw if bpw is not None else "NOT_RUN":.6f}**
- Detail: `{bpw_detail if bpw is not None else {}}`
- Kept high-precision (16-bit accounting): embeddings, layer norms, lm_head, biases
- Simulation note: weights remain float tensors; bpw is packing estimate, not kernel-packed storage

## 7. Falsifier outcome

**{falsifier}**

Issues: {issues if issues else "none"}

Checks: same n_tokens; rotation flag matches arm; n_modules>0; raw logs present; A1/A3 seeds not collapsed.

## 8. What this does NOT prove

- No out-of-sample selector power for κ_proxy
- No 7B multi-model campaign
- No SOTA W3 claim vs QuIP# / AQLM / QTIP
- CPU 135M proxy may not transfer to 0.5B–7B GPU regimes

## 9. Next actions

{next_actions}

## 10. Ledger pointer

- Path: `kappaq_mini/experiments_log.jsonl`
- Run IDs: {[r['id'] for r in runs]}
- Row count (all phases): see file
- Generated: {utc()}
"""
    # fix f-string issue with conditional format
    if bpw is None:
        report = report.replace("**NOT_RUN:.6f**", "**NOT_RUN**")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")

    STATUS.write_text(
        f"""# Status report — C-MINI-LADDER-v1

**Updated:** {utc()}

| Phase | Status |
| --- | --- |
| A Setup | DONE |
| B Main runs | DONE |
| C Analysis | DONE |
| D Report | DONE |

Verdict: **{verdict}** | Falsifier: **{falsifier}**
""",
        encoding="utf-8",
    )

    append_ledger(
        {
            "id": "C-MINI-LADDER-v1-ANALYSIS",
            "phase": "analysis",
            "timestamp_utc": utc(),
            "arm": None,
            "seed": None,
            "model": model,
            "hardware": hardware,
            "config": {"predictions": {"P1": p1, "P2": p2, "P3": p3, "p2_effect_ok": p2_effect_ok}},
            "bpw_effective_est": bpw,
            "metrics": {a: {"mean": table[a]["mean"], "std": table[a]["std"], "ppls": table[a]["ppls"]} for a in arms},
            "raw_log_path": "reports/FINAL_REPORT.md",
            "verdict": verdict,
            "notes": f"falsifier={falsifier}; issues={issues}",
        }
    )
    append_ledger(
        {
            "id": "C-MINI-LADDER-v1-FALSIFY",
            "phase": "falsify",
            "timestamp_utc": utc(),
            "arm": None,
            "seed": None,
            "model": model,
            "hardware": hardware,
            "config": {},
            "bpw_effective_est": bpw,
            "metrics": {"n_tokens_set": list(set(token_counts))},
            "raw_log_path": "reports/FINAL_REPORT.md",
            "verdict": "PASS" if falsifier == "PASS" else "INVALID",
            "notes": "; ".join(issues) if issues else "all checks passed",
        }
    )

    print(json.dumps({"verdict": verdict, "falsifier": falsifier, "P1": p1, "P2": p2, "P3": p3}, indent=2))
    return 0 if falsifier == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
