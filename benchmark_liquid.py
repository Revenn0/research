#!/usr/bin/env python3
"""Benchmarks reais: CIFAR-10 (vs SE/CBAM) + LM Liquid-style gated conv (vs attention)."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# CIFAR-10 models (protocolo ATLAS_CIFAR10_Colab)
# ---------------------------------------------------------------------------


class LocalBlend(nn.Module):
    def __init__(self, channels: int, kernel: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw(x)
        gate = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * x


class MultiScaleBlend(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        for m in (self.dw3, self.dw5):
            nn.init.dirac_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = 0.5 * self.dw3(x) + 0.5 * self.dw5(x)
        gate = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * x


class CosineGateBlend(nn.Module):
    """Gate por cosseno espacial — análogo ao gated short conv da Liquid AI (visão 2D)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw(x)
        dot = (x * local).sum(dim=(2, 3), keepdim=True)
        norm = x.norm(dim=(2, 3), keepdim=True) * local.norm(dim=(2, 3), keepdim=True) + 1e-6
        gate = torch.sigmoid(dot / norm)
        return gate * local + (1.0 - gate) * x


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


def make_mixing(name: str, ch: int) -> nn.Module:
    return {
        "local_blend": LocalBlend(ch),
        "multi_scale_blend": MultiScaleBlend(ch),
        "cosine_gate_blend": CosineGateBlend(ch),
    }.get(name, nn.Identity())


def make_attention(name: str, ch: int) -> nn.Module:
    return {"se": SEBlock(ch)}.get(name, nn.Identity())


class CIFARSmallCNN(nn.Module):
    def __init__(self, mixing: str = "none", attention: str = "none", width_mult: float = 2.0, hidden_dim: int = 256) -> None:
        super().__init__()
        c1, c2 = int(32 * width_mult), int(64 * width_mult)
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.mix1 = make_mixing(mixing, c1)
        self.mix2 = make_mixing(mixing, c2)
        self.att1 = make_attention(attention, c1)
        self.att2 = make_attention(attention, c2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(c2 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _block(self, conv, mix, att, x):
        h = self.act(conv(x))
        h = mix(h)
        h = att(h)
        return self.pool(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._block(self.conv1, self.mix1, self.att1, x)
        x = self._block(self.conv2, self.mix2, self.att2, x)
        x = self.dropout(self.act(self.fc1(x.flatten(1))))
        return self.fc2(x)


def lr_at_step(base_lr: float, step: int, total: int, warmup: int, scheduler: str) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if scheduler == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return base_lr * 0.5 * (1 + math.cos(math.pi * p))
    return base_lr


def load_cifar_cpu(data_dir: str = "./data") -> tuple[TensorDataset, TensorDataset]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    val_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    val_x = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    val_y = torch.tensor([val_ds[i][1] for i in range(len(val_ds))])
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)


@torch.no_grad()
def eval_cifar(model: nn.Module, val_x: torch.Tensor, val_y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    correct = 0
    bs = 512
    for i in range(0, len(val_y), bs):
        x, y = val_x[i : i + bs].to(device), val_y[i : i + bs].to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(val_y)


def train_cifar(
    name: str,
    *,
    mixing: str = "none",
    attention: str = "none",
    label_smoothing: float = 0.0,
    warmup: int = 0,
    scheduler: str = "none",
    steps: int = 400,
    lr: float = 2e-3,
    seed: int = 1000,
    batch: int = 128,
    device: str = "cpu",
) -> dict:
    torch.set_num_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_ds, val_ds = load_cifar_cpu()
    train_x, train_y = train_ds.tensors[0], train_ds.tensors[1]
    val_x, val_y = val_ds.tensors[0], val_ds.tensors[1]

    model = CIFARSmallCNN(mixing=mixing, attention=attention).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best = 0.0
    t0 = time.time()
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(train_y), (batch,))
        x, y = train_x[idx].to(dev), train_y[idx].to(dev)
        for pg in opt.param_groups:
            pg["lr"] = lr_at_step(lr, step, steps, warmup, scheduler)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y, label_smoothing=label_smoothing)
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 4) == 0 or step == steps - 1:
            best = max(best, eval_cifar(model, val_x, val_y, dev))
            model.train()
    wall = time.time() - t0
    return {
        "benchmark": "CIFAR-10",
        "name": name,
        "seed": seed,
        "best_val_acc": best,
        "steps": steps,
        "wall_time_s": wall,
        "steps_per_sec": steps / max(wall, 1e-9),
        "mixing": mixing,
        "attention": attention,
    }


# ---------------------------------------------------------------------------
# LM: Liquid-style Gated Short Conv (LFM2 paper) vs global attention
# ---------------------------------------------------------------------------


class LiquidGatedShortConv(nn.Module):
    """Gated short conv causal — inspirado em LFM2 (Liquid AI)."""

    def __init__(self, n_embd: int, kernel: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.kernel = kernel
        self.in_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dw = nn.Conv1d(n_embd, n_embd, kernel, groups=n_embd, bias=False)
        nn.init.dirac_(self.dw.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        b_gate, c_gate, h = self.in_proj(x).split(c, dim=-1)
        u = b_gate * torch.silu(h)
        u = F.pad(u.transpose(1, 2), (self.kernel - 1, 0))
        y = self.dw(u).transpose(1, 2)
        out = c_gate * y
        return self.dropout(self.out_proj(out))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class LMBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, mixing: str) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if mixing == "global-attn":
            self.mixer = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        elif mixing == "liquid-gsc":
            self.mixer = LiquidGatedShortConv(n_embd, kernel=4, dropout=dropout)
        elif mixing == "hybrid-liquid":
            self.mixer = LiquidGatedShortConv(n_embd, kernel=4, dropout=dropout)
        else:
            raise ValueError(mixing)
        self.mixing = mixing
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniCharLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 8,
        n_head: int = 4,
        n_embd: int = 384,
        block_size: int = 128,
        dropout: float = 0.1,
        mixing: str = "global-attn",
        attn_layers: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        blocks = []
        for i in range(n_layer):
            m = "global-attn" if i in attn_layers else mixing
            if mixing == "hybrid-liquid" and i not in attn_layers:
                m = "liquid-gsc"
            blocks.append(LMBlock(n_embd, n_head, block_size, dropout, m))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_shakespeare(path: str = "atlas/data/tinyshakespeare.txt") -> tuple[torch.Tensor, torch.Tensor, int]:
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], len(chars)


@torch.no_grad()
def eval_lm(model: nn.Module, data: torch.Tensor, block_size: int, batch: int, device: torch.device) -> float:
    model.eval()
    losses = []
    n = min(20, max(1, (len(data) - block_size - 1) // batch))
    for start in range(0, n * batch, batch):
        idx = torch.arange(start, start + batch)
        x = torch.stack([data[i : i + block_size] for i in idx]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(sum(losses) / len(losses))


def train_lm(
    name: str,
    mixing: str,
    *,
    attn_layers: tuple[int, ...] = (),
    steps: int = 3000,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    torch.set_num_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    train, val, vocab = load_shakespeare()
    cfg = dict(n_layer=8, n_head=4, n_embd=384, block_size=128, dropout=0.1)
    model = MiniCharLM(vocab, mixing=mixing, attn_layers=attn_layers, **cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    best = float("inf")
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        max_start = len(train) - cfg["block_size"] - 1
        ix = torch.randint(max_start, (32,))
        x = torch.stack([train[i : i + cfg["block_size"]] for i in ix]).to(dev)
        y = torch.stack([train[i + 1 : i + cfg["block_size"] + 1] for i in ix]).to(dev)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps:
            v = eval_lm(model, val, cfg["block_size"], 32, dev)
            best = min(best, v)
    wall = time.time() - t0
    return {
        "benchmark": "Shakespeare-char-LM",
        "name": name,
        "mixing": mixing,
        "attn_layers": list(attn_layers),
        "params_m": n_params / 1e6,
        "seed": seed,
        "best_val_loss": best,
        "steps": steps,
        "wall_time_s": wall,
        "steps_per_sec": steps / max(wall, 1e-9),
    }


def run_all() -> list[dict]:
    results: list[dict] = []
    print("=== CIFAR-10 (benchmark real vs SE-Net / multi_scale) ===")
    cifar_configs = {
        "baseline_ls_wc": dict(mixing="none", attention="none", label_smoothing=0.1, warmup=200, scheduler="cosine"),
        "multi_scale_ls_wc": dict(mixing="multi_scale_blend", label_smoothing=0.1, warmup=200, scheduler="cosine"),
        "cosine_gate_ls_wc": dict(mixing="cosine_gate_blend", label_smoothing=0.1, warmup=200, scheduler="cosine"),
        "se_ls_wc": dict(attention="se", label_smoothing=0.1, warmup=200, scheduler="cosine"),
    }
    for cname, kw in cifar_configs.items():
        accs = []
        for seed in (1000, 1001, 1002):
            r = train_cifar(cname, seed=seed, steps=400, **kw)
            accs.append(r["best_val_acc"])
            print(f"  {cname} seed{seed}: {r['best_val_acc']*100:.2f}%")
        results.append(
            {
                "benchmark": "CIFAR-10",
                "name": cname,
                "best_val_acc_mean": mean(accs),
                "best_val_acc_std": stdev(accs) if len(accs) > 1 else 0.0,
                "seeds": [1000, 1001, 1002],
                "reference": "colab_cifar10_report.md",
            }
        )

    print("\n=== LM Shakespeare (~12M) Liquid gated conv vs attention ===")
    lm_configs = [
        ("global-attn", "global-attn", ()),
        ("liquid-gsc-full", "liquid-gsc", ()),
        ("hybrid-liquid", "hybrid-liquid", (2, 5)),  # 2 attn / 8 layers ~ LFM2 ratio
    ]
    for name, mixing, attn_layers in lm_configs:
        r = train_lm(name, mixing, attn_layers=attn_layers, steps=3000, seed=42)
        results.append(r)
        print(f"  {name}: val_loss={r['best_val_loss']:.4f}  {r['steps_per_sec']:.2f} steps/s")

    out = Path("benchmark_liquid_results.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def summarize(results: list[dict]) -> dict:
    cifar = [r for r in results if r.get("benchmark") == "CIFAR-10"]
    base = next(r for r in cifar if r["name"] == "baseline_ls_wc")
    ms = next(r for r in cifar if r["name"] == "multi_scale_ls_wc")
    cg = next(r for r in cifar if r["name"] == "cosine_gate_ls_wc")
    se = next(r for r in cifar if r["name"] == "se_ls_wc")
    base_acc = base["best_val_acc_mean"]
    positives = []
    d_cg_base = (cg["best_val_acc_mean"] - base_acc) * 100
    d_cg_se = (cg["best_val_acc_mean"] - se["best_val_acc_mean"]) * 100
    d_cg_ms = (cg["best_val_acc_mean"] - ms["best_val_acc_mean"]) * 100
    d_ms_se = (ms["best_val_acc_mean"] - se["best_val_acc_mean"]) * 100
    if d_cg_se >= 0.3:
        positives.append(f"CIFAR cosine_gate bate SE-Net+stack: {d_cg_se:+.2f}pp")
    if d_cg_ms >= 0.0:
        positives.append(f"CIFAR cosine_gate >= multi_scale: {d_cg_ms:+.2f}pp")
    if d_ms_se >= 0.3:
        positives.append(f"CIFAR multi_scale bate SE-Net (repro): {d_ms_se:+.2f}pp")
    if d_cg_base >= 0.8:
        positives.append(f"CIFAR cosine_gate vs baseline: {d_cg_base:+.2f}pp (REVOLUCIONÁRIA)")

    lm = [r for r in results if r.get("benchmark") == "Shakespeare-char-LM"]
    if lm:
        attn = next((r for r in lm if r["name"] == "global-attn"), None)
        best = min(lm, key=lambda r: r["best_val_loss"])
        if attn and best["best_val_loss"] < attn["best_val_loss"] - 0.05:
            positives.append(
                f"LM {best['name']} val_loss {best['best_val_loss']:.4f} vs attn {attn['best_val_loss']:.4f}"
            )
        liquid = next((r for r in lm if "liquid" in r["name"]), None)
        if attn and liquid and liquid["steps_per_sec"] > attn["steps_per_sec"] * 1.1 and liquid["best_val_loss"] <= attn["best_val_loss"] + 0.05:
            positives.append(f"LM liquid speedup {liquid['steps_per_sec']/attn['steps_per_sec']:.2f}x com loss comparável")

    return {
        "positives": positives,
        "cifar": {
            "baseline": base_acc * 100,
            "cosine_gate": cg["best_val_acc_mean"] * 100,
            "multi_scale": ms["best_val_acc_mean"] * 100,
            "se_ls_wc": se["best_val_acc_mean"] * 100,
        },
    }


if __name__ == "__main__":
    results = run_all()
    summary = summarize(results)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    Path("benchmark_liquid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["positives"]:
        sys.exit(1)
