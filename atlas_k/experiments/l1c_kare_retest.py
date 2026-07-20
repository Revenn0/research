"""
ATLAS-K L1c — MQAR proxy repair + KARE retest (5 seeds)
Goal: get softmax attention >90% (proxy validity), then retest KARE vs GDN-2.
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


def append_ledger(entry: Dict[str, Any]) -> None:
    entry = dict(entry)
    entry["logged_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["code_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def set_seed(s: int) -> None:
    torch.manual_seed(s)


def make_mqar(batch, n_pairs, n_queries, n_keys, n_vals, seed, device):
    """Keys in 1..n_keys, values in n_keys+1..n_keys+n_vals (exclusive ranges)."""
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
    y = torch.full((batch, x.size(1)), -100, device=device, dtype=torch.long)
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
        a = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1)
        return self.o((a @ v).transpose(1, 2).contiguous().view(B, T, D))


class GDN2(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(d, 5 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        k, v, b, w, a = self.proj(x).split(self.d, -1)
        k = F.normalize(k, dim=-1)
        b, w, a = torch.sigmoid(b), torch.sigmoid(w), torch.sigmoid(a)
        S = x.new_zeros(B, self.d, self.d)
        ys = []
        for t in range(T):
            kt, vt, bt, wt, at = k[:, t], v[:, t], b[:, t], w[:, t], a[:, t]
            S = S * at.unsqueeze(-1)
            erase = bt * torch.einsum("bij,bj->bi", S, kt)
            S = S - torch.einsum("bi,bj->bij", erase, kt) + torch.einsum("bi,bj->bij", wt * vt, kt)
            ys.append(torch.einsum("bij,bj->bi", S, kt))
        return self.out(torch.stack(ys, 1))


class KARE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(d, 7 * d, bias=False)  # k v b w a g we
        self.out = nn.Linear(2 * d, d, bias=False)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, T, _ = x.shape
        k, v, b, w, a, g, we = self.proj(x).split(self.d, -1)
        k = F.normalize(k, dim=-1)
        b, w, a, g, we = map(torch.sigmoid, (b, w, a, g, we))
        S = x.new_zeros(B, self.d, self.d)
        E = x.new_zeros(B, self.d, self.d)
        ys = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            bt, wt, at, gt, wet = b[:, t], w[:, t], a[:, t], g[:, t], we[:, t]
            S = S * at.unsqueeze(-1)
            erase = bt * torch.einsum("bij,bj->bi", S, kt)
            S = S - torch.einsum("bi,bj->bij", erase, kt) + torch.einsum("bi,bj->bij", wt * vt, kt)
            E = E * gt.unsqueeze(-1) + torch.einsum("bi,bj->bij", wet * erase, kt)
            ys.append(torch.cat([
                torch.einsum("bij,bj->bi", S, kt),
                self.beta * torch.einsum("bij,bj->bi", E, kt),
            ], -1))
        return self.out(torch.stack(ys, 1))


class LM(nn.Module):
    def __init__(self, vocab, d, kind, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        blocks = []
        for _ in range(layers):
            if kind == "attn":
                mix = Attn(d)
            elif kind == "gdn2":
                mix = GDN2(d)
            elif kind == "kare":
                mix = KARE(d)
            elif kind == "kare_no_echo":
                mix = KARE(d)
            else:
                raise ValueError(kind)
            blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "mix": mix,
                "n2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d, vocab, bias=False)
        if kind == "kare_no_echo":
            for b in self.blocks:
                b["mix"].beta.data.zero_()
                b["mix"].beta.requires_grad_(False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = h + b["mix"](b["n1"](h))
            h = h + b["ff"](b["n2"](h))
        return self.head(h)


@dataclass
class Cfg:
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    steps: int = 1500
    batch: int = 64
    n_pairs: int = 8
    n_queries: int = 4
    n_keys: int = 16
    n_vals: int = 16
    d: int = 64
    layers: int = 2
    lr: float = 1e-3
    pass_margin: float = 0.05


def run_one(kind, seed, cfg: Cfg, device):
    set_seed(seed)
    vocab = cfg.n_keys + cfg.n_vals + 1
    model = LM(vocab, cfg.d, kind, cfg.layers).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
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
            x, y, tgt = make_mqar(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.n_keys, cfg.n_vals, 50_000_000 + seed * 1000 + e, device)
            pred = model(x)[:, -cfg.n_queries:].argmax(-1)
            correct += int((pred == tgt).sum().item())
            total += int(tgt.numel())
    return {
        "kind": kind,
        "seed": seed,
        "acc": correct / max(total, 1),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sec": time.time() - t0,
    }


def main():
    device = torch.device("cpu")
    torch.set_num_threads(4)
    cfg = Cfg()
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": "H-ARCH-KARE",
        "variant": "l1c_proxy_repair",
        "config": asdict(cfg),
        "prediction": "attn mean>=0.90; KARE-GDN2 >=0.05 and >2σ; KARE - no_echo >=0.03",
        "kill_criterion": "proxy invalid or KARE fails margins",
    })
    kinds = ["attn", "gdn2", "kare", "kare_no_echo"]
    runs = []
    for kind in kinds:
        for seed in cfg.seeds:
            r = run_one(kind, seed, cfg, device)
            runs.append(r)
            append_ledger({"type": "result", "gate": "L1", "hyp_id": "H-ARCH-KARE", "variant": "l1c", "run": r})
            print(json.dumps(r), flush=True)

    summary = {}
    for kind in kinds:
        xs = [r["acc"] for r in runs if r["kind"] == kind]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        summary[kind] = {"mean": m, "std": sd, "accs": xs}

    attn_ok = summary["attn"]["mean"] >= 0.90
    delta = summary["kare"]["mean"] - summary["gdn2"]["mean"]
    pooled = math.sqrt(0.5 * (summary["kare"]["std"] ** 2 + summary["gdn2"]["std"] ** 2))
    echo = summary["kare"]["mean"] - summary["kare_no_echo"]["mean"]
    verdict = "PASS_L1" if (attn_ok and delta >= cfg.pass_margin and delta > 2 * pooled and echo >= 0.03) else "FAIL_L1"
    out = {"summary": summary, "verdict": verdict, "attn_ok": attn_ok, "delta": delta, "two_sigma": 2 * pooled, "echo_gap": echo, "runs": runs}
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": "H-ARCH-KARE", "variant": "l1c", **{k: out[k] for k in out if k != "runs"}})
    (RESULTS / "arch_kare_l1c.json").write_text(json.dumps(out, indent=2))
    print("=== L1c SUMMARY ===")
    print(json.dumps({k: out[k] for k in out if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
