#!/usr/bin/env python3
"""Run one KappaQ mini-ladder arm × seed job."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpw import estimate_bpw, list_dense_modules  # noqa: E402
from eval_ppl import build_eval_batches, eval_perplexity, load_wikitext_text  # noqa: E402
from quant import quantize_weight  # noqa: E402
from rotate import make_orthogonal, rotate_weight, unrotate_weight  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_ledger(row: Dict[str, Any], ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_arm(model: nn.Module, arm: str, seed: int, group_size: int) -> Dict[str, Any]:
    """In-place quantize dense projections according to arm definition."""
    info: Dict[str, Any] = {"arm": arm, "modules": [], "rotation_applied": False}
    rotation_cache: Dict[int, torch.Tensor] = {}

    use_rot = arm in ("A1", "A3", "rot", "rot+RTN", "rot+mse")
    use_mse = arm in ("A2", "A3", "mse", "rot+mse")
    method = "mse" if use_mse else "rtn"

    for name, mod in list_dense_modules(model):
        w = mod.weight.data
        in_f = w.shape[1]
        w_work = w
        R = None
        if use_rot:
            if in_f not in rotation_cache:
                rotation_cache[in_f] = make_orthogonal(in_f, seed=seed, device=w.device)
            R = rotation_cache[in_f]
            w_work = rotate_weight(w, R)
            info["rotation_applied"] = True
        q, meta = quantize_weight(w_work, method=method, group_size=group_size)
        if use_rot and R is not None:
            q = unrotate_weight(q, R)
        mod.weight.data.copy_(q)
        info["modules"].append({"name": name, "meta": meta, "shape": list(w.shape)})
    info["n_modules"] = len(info["modules"])
    info["method"] = method
    info["use_rot"] = use_rot
    return info


def peak_mem_mb() -> Optional[float]:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["A0", "A1", "A2", "A3"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-sequences", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--smoke", action="store_true", help="2 sequences only")
    p.add_argument("--ledger", default=str(ROOT / "experiments_log.jsonl"))
    p.add_argument("--skip-ledger", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n_sequences = 2
        args.max_length = 128

    arm_names = {"A0": "RTN", "A1": "rot+RTN", "A2": "mse", "A3": "rot+mse"}
    run_id = f"C-MINI-LADDER-v1-{args.arm}-s{args.seed}"
    if args.smoke:
        run_id = f"SMOKE-{run_id}"

    out_dir = ROOT / "runs" / args.arm / f"seed{args.seed}"
    if args.smoke:
        out_dir = ROOT / "runs" / "_smoke" / args.arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.json"
    raw_txt = out_dir / "raw.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )

    def log(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with raw_txt.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    result: Dict[str, Any] = {
        "id": run_id,
        "phase": "run",
        "timestamp_utc": utc_now(),
        "arm": args.arm,
        "arm_name": arm_names[args.arm],
        "seed": args.seed,
        "model": args.model,
        "hardware": hardware,
        "config": {
            "bits": 3,
            "group_size": args.group_size,
            "rotation": args.arm in ("A1", "A3"),
            "clip": "mse_grid" if args.arm in ("A2", "A3") else "rtn_maxabs",
            "n_sequences": args.n_sequences,
            "max_length": args.max_length,
            "symmetric": True,
            "smoke": args.smoke,
        },
        "bpw_effective_est": None,
        "metrics": {},
        "raw_log_path": str(log_path.relative_to(ROOT)),
        "verdict": "PENDING",
        "notes": "",
    }

    t_job0 = time.perf_counter()
    try:
        log(f"START {run_id} device={device} model={args.model}")
        set_global_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        log("loading tokenizer/model...")
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if device.type == "cpu" else torch.float16
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.to(device)
        model.eval()

        bpw = estimate_bpw(model, bits_dense=3, group_size=args.group_size)
        result["bpw_effective_est"] = bpw["bpw_effective_est"]
        result["bpw_detail"] = bpw
        log(f"bpw_effective_est={bpw['bpw_effective_est']:.4f} dense_frac={bpw['fraction_dense']:.4f}")

        # FP16 baseline check on smoke only (optional) — always apply arm
        log(f"applying arm {args.arm}...")
        arm_info = apply_arm(model, args.arm, seed=args.seed, group_size=args.group_size)
        result["arm_info_summary"] = {
            "n_modules": arm_info["n_modules"],
            "method": arm_info["method"],
            "use_rot": arm_info["use_rot"],
            "rotation_applied": arm_info["rotation_applied"],
        }
        log(f"quantized modules={arm_info['n_modules']} rot={arm_info['rotation_applied']}")

        text, source, chash = load_wikitext_text()
        batches, n_tok_raw = build_eval_batches(
            tok, text, n_sequences=args.n_sequences, max_length=args.max_length
        )
        log(f"eval corpus={source} hash={chash} chunks={len(batches)} raw_tokens={n_tok_raw}")

        ppl, n_tokens, wall_s = eval_perplexity(model, batches, device)
        peak = peak_mem_mb()
        job_wall = time.perf_counter() - t_job0
        result["metrics"] = {
            "ppl": ppl,
            "n_tokens": n_tokens,
            "wall_s": wall_s,
            "job_wall_s": job_wall,
            "peak_vram_mb": peak,
            "corpus_source": source,
            "corpus_hash": chash,
            "n_sequences": len(batches),
            "max_length": args.max_length,
        }
        result["verdict"] = "PASS" if ppl == ppl and ppl > 0 else "FAIL"
        result["notes"] = f"finite_ppl={ppl == ppl}"
        log(f"DONE ppl={ppl:.6f} n_tokens={n_tokens} wall_s={wall_s:.2f} job_wall_s={job_wall:.2f}")

    except Exception as e:
        result["verdict"] = "FAIL"
        result["notes"] = f"exception: {e}"
        result["metrics"] = {"ppl": None, "n_tokens": 0, "wall_s": 0.0}
        log("ERROR:\n" + traceback.format_exc())
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        if not args.skip_ledger:
            append_ledger(result, Path(args.ledger))
        return 1

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if not args.skip_ledger:
        append_ledger(result, Path(args.ledger))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
