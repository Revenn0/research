"""
ATLAS-K L1e — Fixed MQAR protocol + Conflict-Protected Erase (CPE)

Proxy fix:
  - sinusoidal positional encodings
  - value-shift: at t, bind k_t with v_{t} where v is computed from x, but we also
    feed a shifted residual so value tokens carry previous key in stream
  - more standard: use k from x_t and v from x_t, BUT concatenate [x_t; x_{t-1}] into mixer input

CPE (candidate):
  read = S k
  conflict = sigmoid(α * <normalize(v), normalize(read)>)  # high if overwriting similar
  # Protect when NEW content conflicts with a DIFFERENT stored value:
  # protect = sigmoid(β * ||read|| * (1 - cos(v, read)))
  erase = b * (1 - protect) * read
  ... standard write ...

Nearest neighbor: GDN-2. Delta: content-adaptive protection of erase based on
value-read disagreement energy.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
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


def sinusoid_pe(T, d, device):
    pe = torch.zeros(T, d, device=device)
    pos = torch.arange(T, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def make_mqar(batch, n_pairs, n_queries, n_keys, n_vals, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    keys = torch.randint(1, n_keys + 1, (batch, n_pairs), generator=g)
    vals = torch.randint(n_keys + 1, n_keys + n_vals + 1, (batch, n_pairs), generator=g)
    q_idx = torch.randint(0, n_pairs, (batch, n_queries), generator=g)
    queries = torch.gather(keys, 1, q_idx)
    targets = torch.gather(vals, 1, q_idx)
    seq = []
    for b in range(batch):
        toks = []
        for i in range(n_pairs):
            toks += [int(keys[b, i]), int(vals[b, i])]
        toks += queries[b].tolist()
        seq.append(toks)
    x = torch.tensor(seq, device=device)
    y = torch.full((batch, x.size(1)), -100, dtype=torch.long, device=device)
    y[:, -n_queries:] = targets.to(device)
    return x, y, targets.to(device)


class Attn(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        # causal mask so queries attend only to past key-value pairs
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, -1e9)
        att = F.softmax(att, dim=-1)
        return self.o((att @ v).transpose(1, 2).contiguous().view(B, T, D))


class GDN2(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        # input is concat(x_t, x_{t-1}) => 2d
        self.proj = nn.Linear(2 * d, 5 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        inp = torch.cat([x, prev], dim=-1)
        k, v, b, w, a = self.proj(inp).split(self.d, -1)
        k = F.normalize(k, dim=-1)
        b, w, a = map(torch.sigmoid, (b, w, a))
        S = x.new_zeros(B, self.d, self.d)
        ys = []
        for t in range(T):
            kt, vt, bt, wt, at = k[:, t], v[:, t], b[:, t], w[:, t], a[:, t]
            S = S * at.unsqueeze(-1)
            erase = bt * torch.einsum("bij,bj->bi", S, kt)
            S = S - torch.einsum("bi,bj->bij", erase, kt) + torch.einsum("bi,bj->bij", wt * vt, kt)
            ys.append(torch.einsum("bij,bj->bi", S, kt))
        return self.out(torch.stack(ys, 1))


class CPE(nn.Module):
    """Conflict-Protected Erase — GDN2 + protect gate on erase."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(2 * d, 5 * d, bias=False)
        self.protect_logit_scale = nn.Parameter(torch.tensor(2.0))
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        inp = torch.cat([x, prev], dim=-1)
        k, v, b, w, a = self.proj(inp).split(self.d, -1)
        k = F.normalize(k, dim=-1)
        b, w, a = map(torch.sigmoid, (b, w, a))
        S = x.new_zeros(B, self.d, self.d)
        ys = []
        for t in range(T):
            kt, vt, bt, wt, at = k[:, t], v[:, t], b[:, t], w[:, t], a[:, t]
            S = S * at.unsqueeze(-1)
            read = torch.einsum("bij,bj->bi", S, kt)
            # protect when read is strong AND disagrees with new v
            cos = F.cosine_similarity(vt, read, dim=-1).unsqueeze(-1)
            strength = read.norm(dim=-1, keepdim=True)
            protect = torch.sigmoid(self.protect_logit_scale * strength * (1.0 - cos))
            erase = bt * (1.0 - protect) * read
            S = S - torch.einsum("bi,bj->bij", erase, kt) + torch.einsum("bi,bj->bij", wt * vt, kt)
            ys.append(torch.einsum("bij,bj->bi", S, kt))
        return self.out(torch.stack(ys, 1))


class LM(nn.Module):
    def __init__(self, vocab, d, kind, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.kind = kind
        blocks = []
        for _ in range(layers):
            if kind == "attn":
                mix = Attn(d)
            elif kind == "gdn2":
                mix = GDN2(d)
            elif kind == "cpe":
                mix = CPE(d)
            else:
                raise ValueError(kind)
            blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "mix": mix,
                "n2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + sinusoid_pe(T, h_dim := self.emb.embedding_dim, x.device)
        for b in self.blocks:
            h = h + b["mix"](b["n1"](h))
            h = h + b["ff"](b["n2"](h))
        return self.head(h)


@dataclass
class Cfg:
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    steps: int = 800
    batch: int = 64
    n_pairs: int = 8
    n_queries: int = 4
    n_keys: int = 16
    n_vals: int = 16
    d: int = 64
    layers: int = 2
    lr: float = 2e-3
    pass_margin: float = 0.05


def run_one(kind, seed, cfg, device):
    set_seed(seed)
    vocab = cfg.n_keys + cfg.n_vals + 1
    model = LM(vocab, cfg.d, kind, cfg.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    t0 = time.time()
    for step in range(cfg.steps):
        x, y, _ = make_mqar(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.n_keys, cfg.n_vals, seed * 100000 + step, device)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for e in range(40):
            x, y, tgt = make_mqar(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.n_keys, cfg.n_vals, 70_000_000 + seed * 1000 + e, device)
            pred = model(x)[:, -cfg.n_queries:].argmax(-1)
            correct += int((pred == tgt).sum().item())
            total += int(tgt.numel())
    return {"kind": kind, "seed": seed, "acc": correct / max(total, 1), "params": sum(p.numel() for p in model.parameters()), "sec": time.time() - t0}


def main():
    device = torch.device("cpu")
    torch.set_num_threads(4)
    cfg = Cfg()
    append_ledger({"type": "preregister", "gate": "L1", "hyp_id": "H-ARCH-CPE", "config": asdict(cfg),
                   "prediction": "attn>=0.85; CPE-GDN2>=0.05 and >2σ",
                   "kill_criterion": "proxy invalid or CPE fails margin"})
    kinds = ["attn", "gdn2", "cpe"]
    runs = []
    for kind in kinds:
        for seed in cfg.seeds:
            r = run_one(kind, seed, cfg, device)
            runs.append(r)
            append_ledger({"type": "result", "gate": "L1", "hyp_id": "H-ARCH-CPE", "run": r})
            print(json.dumps(r), flush=True)
    summary = {}
    for kind in kinds:
        xs = [r["acc"] for r in runs if r["kind"] == kind]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        summary[kind] = {"mean": m, "std": sd, "accs": xs}
    attn_ok = summary["attn"]["mean"] >= 0.85
    delta = summary["cpe"]["mean"] - summary["gdn2"]["mean"]
    pooled = math.sqrt(0.5 * (summary["cpe"]["std"] ** 2 + summary["gdn2"]["std"] ** 2))
    verdict = "PASS_L1" if (attn_ok and delta >= cfg.pass_margin and delta > 2 * pooled) else "FAIL_L1"
    out = {"summary": summary, "verdict": verdict, "attn_ok": attn_ok, "delta": delta, "two_sigma": 2 * pooled, "runs": runs}
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": "H-ARCH-CPE", "verdict": verdict,
                   "summary": summary, "attn_ok": attn_ok, "delta": delta})
    (RESULTS / "arch_cpe_l1e.json").write_text(json.dumps(out, indent=2))
    print("=== L1e SUMMARY ===")
    print(json.dumps({k: out[k] for k in out if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
