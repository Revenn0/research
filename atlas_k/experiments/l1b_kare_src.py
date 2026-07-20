"""
ATLAS-K L1b — Key-Addressable Residual Echo (KARE) + Sign Residual Codebook (SRC)
Second-wave candidates after L1 failures of REM / SRP / SS-Muon.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger" / "ledger.jsonl"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def append_ledger(entry: Dict[str, Any]) -> None:
    entry = dict(entry)
    entry.setdefault("logged_utc", utc_now())
    entry.setdefault("code_hash", code_hash())
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def env_block() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
    }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------- MQAR data ----------------

def make_mqar_batch(batch, n_pairs, n_queries, vocab, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    key_vocab = vocab // 2
    keys = torch.randint(1, key_vocab, (batch, n_pairs), generator=g)
    vals = torch.randint(key_vocab, vocab, (batch, n_pairs), generator=g)
    q_idx = torch.randint(0, n_pairs, (batch, n_queries), generator=g)
    queries = torch.gather(keys, 1, q_idx)
    targets = torch.gather(vals, 1, q_idx)
    seq = []
    for b in range(batch):
        toks = []
        for i in range(n_pairs):
            toks.extend([int(keys[b, i]), int(vals[b, i])])
        toks.extend(queries[b].tolist())
        seq.append(toks)
    x = torch.tensor(seq, dtype=torch.long, device=device)
    y = torch.full((batch, x.size(1)), -100, dtype=torch.long, device=device)
    y[:, -n_queries:] = targets.to(device)
    return x, y, targets.to(device)


# ---------------- Mixers ----------------

class SoftmaxAttentionMixer(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class GatedDeltaMixer(nn.Module):
    """Tied erase/write scalar β + channel decay (GDN/KDA-lite)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.in_proj = nn.Linear(d, 4 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        k, v, beta_l, decay_l = self.in_proj(x).split(self.d, dim=-1)
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(beta_l)
        alpha = torch.sigmoid(decay_l)
        S = x.new_zeros(B, self.d, self.d)
        outs = []
        for t in range(T):
            kt, vt, bt, at = k[:, t], v[:, t], beta[:, t], alpha[:, t]
            S = S * at.unsqueeze(-1)
            read = torch.einsum("bij,bj->bi", S, kt)
            erased = bt * read
            S = S - torch.einsum("bi,bj->bij", erased, kt)
            S = S + torch.einsum("bi,bj->bij", bt * vt, kt)
            outs.append(torch.einsum("bij,bj->bi", S, kt))
        return self.out_proj(torch.stack(outs, 1))


class GatedDeltaNet2Lite(nn.Module):
    """Decoupled channel-wise erase b and write w + channel decay (GDN-2 lite)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.in_proj = nn.Linear(d, 5 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        k, v, b_l, w_l, decay_l = self.in_proj(x).split(self.d, dim=-1)
        k = F.normalize(k, dim=-1)
        b = torch.sigmoid(b_l)
        w = torch.sigmoid(w_l)
        alpha = torch.sigmoid(decay_l)
        S = x.new_zeros(B, self.d, self.d)
        outs = []
        for t in range(T):
            kt, vt, bt, wt, at = k[:, t], v[:, t], b[:, t], w[:, t], alpha[:, t]
            S = S * at.unsqueeze(-1)
            read = torch.einsum("bij,bj->bi", S, kt)
            erase = bt * read
            S = S - torch.einsum("bi,bj->bij", erase, kt)
            S = S + torch.einsum("bi,bj->bij", wt * vt, kt)
            outs.append(torch.einsum("bij,bj->bi", S, kt))
        return self.out_proj(torch.stack(outs, 1))


class KAREMixer(nn.Module):
    """
    Key-Addressable Residual Echo (KARE)

    After delta erase, write the erased content into a parallel echo state E
    that is ALSO addressed by keys (matrix state), with slower decay.

      S ← α⊙S
      erase ← b⊙(S k̂)
      S ← S − erase k̂ᵀ + (w⊙v) k̂ᵀ
      E ← γ⊙E + (w_e ⊙ P(erase)) k̂ᵀ     # key-addressable echo
      y ← S k̂ + β_echo * (E k̂)

    Novelty vs GDN-2: explicit recovery path for erased associations.
    Novelty vs REM (failed): echo is key-addressable, not a pooled vector.
    """

    def __init__(self, d, echo_rank_frac: float = 1.0):
        super().__init__()
        self.d = d
        self.ed = max(8, int(d * echo_rank_frac))
        # k,v,erase,write,decay,echo_decay,echo_write
        self.in_proj = nn.Linear(d, 5 * d + 2 * self.ed, bias=False)
        self.erase_to_echo = nn.Linear(d, self.ed, bias=False)
        self.echo_scale = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Linear(d + self.ed, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        parts = self.in_proj(x)
        k, v, b_l, w_l, decay_l, echo_decay_l, echo_w_l = torch.split(
            parts, [self.d, self.d, self.d, self.d, self.d, self.ed, self.ed], dim=-1
        )
        k = F.normalize(k, dim=-1)
        b = torch.sigmoid(b_l)
        w = torch.sigmoid(w_l)
        alpha = torch.sigmoid(decay_l)
        gamma = torch.sigmoid(echo_decay_l)
        we = torch.sigmoid(echo_w_l)
        S = x.new_zeros(B, self.d, self.d)
        E = x.new_zeros(B, self.ed, self.d)
        outs = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            bt, wt, at = b[:, t], w[:, t], alpha[:, t]
            gt, wet = gamma[:, t], we[:, t]
            S = S * at.unsqueeze(-1)
            read = torch.einsum("bij,bj->bi", S, kt)
            erase = bt * read
            S = S - torch.einsum("bi,bj->bij", erase, kt)
            S = S + torch.einsum("bi,bj->bij", wt * vt, kt)
            echo_vec = wet * self.erase_to_echo(erase)
            E = E * gt.unsqueeze(-1)
            E = E + torch.einsum("bi,bj->bij", echo_vec, kt)
            y_s = torch.einsum("bij,bj->bi", S, kt)
            y_e = torch.einsum("bij,bj->bi", E, kt)
            outs.append(torch.cat([y_s, self.echo_scale * y_e], dim=-1))
        return self.out_proj(torch.stack(outs, 1))


class MixerLM(nn.Module):
    def __init__(self, vocab, d, mixer, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if mixer == "attn":
                m = SoftmaxAttentionMixer(d)
            elif mixer == "gated_delta":
                m = GatedDeltaMixer(d)
            elif mixer == "gdn2":
                m = GatedDeltaNet2Lite(d)
            elif mixer == "kare":
                m = KAREMixer(d)
            elif mixer == "kare_no_echo":
                m = KAREMixer(d)
            else:
                raise ValueError(mixer)
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d),
                "mix": m,
                "ffn_n": nn.LayerNorm(d),
                "ffn": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }))
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.mixer_name = mixer
        if mixer == "kare_no_echo":
            for layer in self.layers:
                layer["mix"].echo_scale.data.zero_()
                layer["mix"].echo_scale.requires_grad_(False)

    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = h + layer["mix"](layer["norm"](h))
            h = h + layer["ffn"](layer["ffn_n"](h))
        return self.head(h)


@dataclass
class ArchCfg:
    hyp_id: str = "H-ARCH-KARE"
    seeds: Tuple[int, ...] = (0, 1, 2)
    steps: int = 600
    batch: int = 64
    n_pairs: int = 8
    n_queries: int = 4
    vocab: int = 48
    d_model: int = 64
    n_layers: int = 2
    lr: float = 3e-3
    pass_margin: float = 0.05


def train_eval(mixer, seed, cfg: ArchCfg, device):
    set_seed(seed)
    model = MixerLM(cfg.vocab, cfg.d_model, mixer, cfg.n_layers).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(cfg.steps):
        x, y, _ = make_mqar_batch(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.vocab, seed * 100000 + step, device)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for e in range(30):
            x, y, targets = make_mqar_batch(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.vocab, 10_000_000 + seed * 1000 + e, device)
            pred = model(x)[:, -cfg.n_queries:].argmax(-1)
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
    return {
        "mixer": mixer,
        "seed": seed,
        "accuracy": correct / max(total, 1),
        "params": count_params(model),
        "wall_clock_s": time.time() - t0,
        "steps": cfg.steps,
    }


def run_kare_l1(device):
    cfg = ArchCfg()
    # also secondary proxy: selective copy
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": cfg.hyp_id,
        "config": asdict(cfg),
        "prediction": "mean(KARE)-mean(GDN2)>=0.05 and >2σ; ablation kare_no_echo gap>=0.03; attention sanity mean>=0.5",
        "kill_criterion": "fails margin vs GDN2 or echo ablation or attention proxy broken",
        "baselines": ["attn", "gated_delta", "gdn2", "kare", "kare_no_echo"],
        "env": env_block(),
    })
    mixers = ["attn", "gated_delta", "gdn2", "kare", "kare_no_echo"]
    runs = []
    for mixer in mixers:
        for seed in cfg.seeds:
            try:
                r = train_eval(mixer, seed, cfg, device)
                r["status"] = "ok"
            except Exception as e:
                r = {"mixer": mixer, "seed": seed, "status": "error", "error": str(e), "trace": traceback.format_exc()}
            runs.append(r)
            append_ledger({"type": "result", "gate": "L1", "hyp_id": cfg.hyp_id, "run": r})
            print(json.dumps({k: r[k] for k in r if k != "trace"}))

    summary = {}
    for mixer in mixers:
        xs = [r["accuracy"] for r in runs if r.get("status") == "ok" and r["mixer"] == mixer]
        mean = sum(xs) / len(xs)
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1))
        summary[mixer] = {"mean_acc": mean, "std_acc": std, "accs": xs, "params": next(r["params"] for r in runs if r.get("mixer") == mixer and r.get("status") == "ok")}

    attn_ok = summary["attn"]["mean_acc"] >= 0.5
    delta = summary["kare"]["mean_acc"] - summary["gdn2"]["mean_acc"]
    pooled = math.sqrt(0.5 * (summary["kare"]["std_acc"] ** 2 + summary["gdn2"]["std_acc"] ** 2))
    echo_gap = summary["kare"]["mean_acc"] - summary["kare_no_echo"]["mean_acc"]
    verdict = "PASS_L1" if (attn_ok and delta >= cfg.pass_margin and delta > 2 * pooled and echo_gap >= 0.03) else "FAIL_L1"
    reason = f"attn_ok={attn_ok} delta={delta:.4f} 2σ={2*pooled:.4f} echo_gap={echo_gap:.4f} summary={summary}"
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": cfg.hyp_id, "verdict": verdict, "reason": reason, "summary": summary})
    out = {"hyp_id": cfg.hyp_id, "summary": summary, "verdict": verdict, "reason": reason, "runs": runs}
    (RESULTS / "arch_kare_l1.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------- Quant: Sign Residual Codebook ----------------

def fake_q(w, bits, axis="row"):
    qmax = 2 ** (bits - 1) - 1
    if w.ndim == 1:
        scale = w.abs().amax().clamp_min(1e-8) / qmax
        return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    if axis == "row":
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    else:
        scale = w.abs().amax().clamp_min(1e-8) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def gptq_with_H(w, bits, H, blocksize=32):
    W = w.clone().float()
    cols = W.size(1)
    hat = torch.zeros_like(W)
    H = H.float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(cols, device=W.device)
    H[diag, diag] += damp
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        Hinv = torch.linalg.pinv(H)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(count):
            w_j = W1[:, j]
            d = Hinv1[j, j].clamp_min(1e-8)
            q_j = fake_q(w_j, bits)
            Q1[:, j] = q_j
            err = (w_j - q_j) / d
            Err1[:, j] = err
            if j + 1 < count:
                W1[:, j + 1 :] -= err.unsqueeze(1) * Hinv1[j, j + 1 :].unsqueeze(0)
        hat[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return hat


def sign_residual_codebook(base_q: torch.Tensor, W: torch.Tensor, nnz_frac: float, residual_bits: int = 1) -> Tuple[torch.Tensor, float]:
    """
    SRC: after base quantizer, keep top-|R| entries of residual with sign (+/- scale).
    For residual_bits=1: store sign only + shared magnitude per row.
    Returns reconstruction and average extra bits per weight.
    """
    R = W - base_q
    rows, cols = W.shape
    k = max(1, int(nnz_frac * cols))
    # per-row top-k by magnitude
    absR = R.abs()
    topv, topi = torch.topk(absR, k=k, dim=1)
    mask = torch.zeros_like(R, dtype=torch.bool)
    mask.scatter_(1, topi, True)
    # row magnitude = mean of selected abs residuals
    mag = (absR * mask).sum(dim=1, keepdim=True) / k
    signs = torch.where(R >= 0, torch.ones_like(R), -torch.ones_like(R))
    residual = torch.zeros_like(R)
    residual[mask] = (signs * mag)[mask]
    # bits: base already counted elsewhere; extra = nnz * 1 bit index? 
    # Honest bit accounting: each selected entry needs log2(cols) index bits + 1 sign bit, amortized per weight.
    # Plus row mag in fp16.
    index_bits = math.log2(cols)
    extra_bits_total = rows * k * (1 + index_bits) + rows * 16
    avg_extra = extra_bits_total / (rows * cols)
    return base_q + residual, avg_extra


def run_src_l1(device):
    hyp = "H-QUANT-SRC"
    seeds = (0, 1, 2, 3, 4)
    rows = cols = 256
    base_bits = 3
    nnz_frac = 0.05  # 5% sparse residual
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": hyp,
        "prediction": "GPTQ3+SRC at its effective bits beats GPTQ at ceil(eff_bits) on MSE and KL",
        "kill_criterion": "does not beat GPTQ_ceil on both MSE and KL",
        "config": {"base_bits": base_bits, "nnz_frac": nnz_frac, "rows": rows},
        "env": env_block(),
    })
    runs = []
    for seed in seeds:
        set_seed(seed)
        U = torch.randn(rows, 32, device=device) / math.sqrt(32)
        V = torch.randn(32, cols, device=device) / math.sqrt(32)
        W = U @ V
        idx = torch.randperm(rows)[: rows // 20]
        W = W + 0.05 * torch.randn_like(W)
        W[idx] += 8.0 * torch.randn(len(idx), cols, device=device)
        X = torch.randn(512, cols, device=device)
        Y = X @ W.T
        H = (X.T @ X) / X.size(0)
        base = gptq_with_H(W, base_bits, H)
        src, extra = sign_residual_codebook(base, W, nnz_frac)
        eff = base_bits + extra
        ceil_bits = int(math.ceil(eff))
        gptq_c = gptq_with_H(W, ceil_bits, H)
        rtn_c = fake_q(W, ceil_bits)

        def met(What):
            mse = float(((What - W) ** 2).mean().item())
            kl = float(F.kl_div(F.log_softmax(X @ What.T, -1), F.softmax(Y, -1), reduction="batchmean").item())
            return {"mse": mse, "kl": kl}

        row = {"seed": seed, "eff_bits": eff, "ceil_bits": ceil_bits, "src": met(src), "gptq_ceil": met(gptq_c), "gptq_base": met(base), "rtn_ceil": met(rtn_c)}
        runs.append(row)
        append_ledger({"type": "result", "gate": "L1", "hyp_id": hyp, "run": row})
        print(json.dumps(row))

    means = {
        "src_mse": sum(r["src"]["mse"] for r in runs) / len(runs),
        "gptq_ceil_mse": sum(r["gptq_ceil"]["mse"] for r in runs) / len(runs),
        "src_kl": sum(r["src"]["kl"] for r in runs) / len(runs),
        "gptq_ceil_kl": sum(r["gptq_ceil"]["kl"] for r in runs) / len(runs),
        "eff_bits_mean": sum(r["eff_bits"] for r in runs) / len(runs),
    }
    verdict = "PASS_L1" if (means["src_mse"] < means["gptq_ceil_mse"] and means["src_kl"] < means["gptq_ceil_kl"]) else "FAIL_L1"
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": hyp, "verdict": verdict, "means": means})
    out = {"hyp_id": hyp, "means": means, "verdict": verdict, "runs": runs}
    (RESULTS / "quant_src_l1.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    append_ledger({"type": "session_start", "suite": "l1b", "env": env_block()})
    out = {"kare": run_kare_l1(device), "src": run_src_l1(device)}
    (RESULTS / "l1b_summary.json").write_text(json.dumps({k: v.get("verdict") for k, v in out.items()}, indent=2))
    print("=== L1b SUMMARY ===")
    print(json.dumps({k: v.get("verdict") for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
