"""
ATLAS-K L1d — Hadamard + Spectral Residual Packing (HSRP)
Hypothesis: random orthogonalization (QuaRot-style) makes residual more low-rank / Gaussian,
so spectral residual packing can beat GPTQ at ceil(effective bits).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger" / "ledger.jsonl"
RESULTS = ROOT / "results"


def append_ledger(e: Dict[str, Any]) -> None:
    e = dict(e)
    e["logged_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    e["code_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    with LEDGER.open("a") as f:
        f.write(json.dumps(e) + "\n")


def set_seed(s):
    torch.manual_seed(s)


def hadamard(n, device):
    """Recursive Hadamard; n must be power of 2."""
    assert n > 0 and (n & (n - 1)) == 0
    H = torch.tensor([[1.0]], device=device)
    while H.size(0) < n:
        a = H
        H = torch.cat([torch.cat([a, a], 1), torch.cat([a, -a], 1)], 0)
    return H / math.sqrt(n)


def fake_q(w, bits, axis="row"):
    qmax = 2 ** (bits - 1) - 1
    if w.ndim == 1:
        scale = w.abs().amax().clamp_min(1e-8) / qmax
        return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    if axis == "row":
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
    else:
        scale = w.abs().amax().clamp_min(1e-8) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def gptq(w, bits, H, block=32):
    W = w.clone().float()
    cols = W.size(1)
    hat = torch.zeros_like(W)
    H = H.float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    d = torch.arange(cols, device=W.device)
    H[d, d] += damp
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        Hinv = torch.linalg.pinv(H)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        c = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(c):
            wj = W1[:, j]
            dj = Hi[j, j].clamp_min(1e-8)
            qj = fake_q(wj, bits)
            Q1[:, j] = qj
            err = (wj - qj) / dj
            Err1[:, j] = err
            if j + 1 < c:
                W1[:, j + 1 :] -= err.unsqueeze(1) * Hi[j, j + 1 :].unsqueeze(0)
        hat[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return hat


def srp_on(base_q, W, rank, rbits):
    R = W - base_q
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
    Uq = fake_q(U, rbits, axis="tensor")
    Vq = fake_q(Vh, rbits, axis="tensor")
    return base_q + (Uq * S.unsqueeze(0)) @ Vq


def eff_bits(rows, cols, base_bits, rank, rbits):
    return (rows * cols * base_bits + rows * rank * rbits + rank * cols * rbits + rank * 16) / (rows * cols)


def main():
    device = torch.device("cpu")
    rows = cols = 256
    base_bits, rank, rbits = 3, 8, 4
    seeds = range(5)
    Hm = hadamard(cols, device)
    eff = eff_bits(rows, cols, base_bits, rank, rbits)
    ceil_b = int(math.ceil(eff))
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": "H-QUANT-HSRP",
        "prediction": "Hadamard-rotate then GPTQ3+SRP beats GPTQ_ceil and RTN_ceil on MSE and KL",
        "kill_criterion": "fails to beat both GPTQ_ceil MSE and KL",
        "eff_bits": eff,
        "ceil_bits": ceil_b,
    })
    runs = []
    for seed in seeds:
        set_seed(seed)
        U = torch.randn(rows, 32) / math.sqrt(32)
        V = torch.randn(32, cols) / math.sqrt(32)
        W = U @ V + 0.05 * torch.randn(rows, cols)
        idx = torch.randperm(rows)[: rows // 20]
        W[idx] += 8 * torch.randn(len(idx), cols)
        # rotate columns (input side): W' = W @ H
        Wr = W @ Hm
        X = torch.randn(512, cols)
        # for rotated system, activations become H^T x if we absorb rotation; for reconstruction of W we compare in original basis
        Y = X @ W.T
        Hr = ((X @ Hm).T @ (X @ Hm)) / X.size(0)  # Hessian in rotated coords
        base = gptq(Wr, base_bits, Hr)
        hsrp_r = srp_on(base, Wr, rank, rbits)
        # map back: What = What_r @ H^T
        hsrp = hsrp_r @ Hm.T
        gptq_c = gptq(W, ceil_b, (X.T @ X) / X.size(0))
        rtn_c = fake_q(W, ceil_b)
        gptq_f = gptq(W, base_bits, (X.T @ X) / X.size(0))
        srp_plain = srp_on(gptq_f, W, rank, rbits)

        def met(What):
            return {
                "mse": float(((What - W) ** 2).mean().item()),
                "kl": float(F.kl_div(F.log_softmax(X @ What.T, -1), F.softmax(Y, -1), reduction="batchmean").item()),
            }

        row = {
            "seed": seed,
            "hsrp": met(hsrp),
            "srp_plain": met(srp_plain),
            "gptq_ceil": met(gptq_c),
            "rtn_ceil": met(rtn_c),
            "gptq_floor": met(gptq_f),
            "eff": eff,
        }
        runs.append(row)
        append_ledger({"type": "result", "gate": "L1", "hyp_id": "H-QUANT-HSRP", "run": row})
        print(json.dumps(row))

    def mean(method, key):
        return sum(r[method][key] for r in runs) / len(runs)

    means = {f"{m}_{k}": mean(m, k) for m in ("hsrp", "srp_plain", "gptq_ceil", "rtn_ceil", "gptq_floor") for k in ("mse", "kl")}
    verdict = "PASS_L1" if (means["hsrp_mse"] < means["gptq_ceil_mse"] and means["hsrp_kl"] < means["gptq_ceil_kl"]) else "FAIL_L1"
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": "H-QUANT-HSRP", "verdict": verdict, "means": means})
    out = {"verdict": verdict, "means": means, "runs": runs, "eff_bits": eff}
    (RESULTS / "quant_hsrp_l1.json").write_text(json.dumps(out, indent=2))
    print("=== HSRP SUMMARY ===")
    print(json.dumps({"verdict": verdict, "means": means}, indent=2))


if __name__ == "__main__":
    main()
