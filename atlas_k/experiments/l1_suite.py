"""
ATLAS-K L1 Experiment Suite
===========================
CPU-executable discriminative proxies for:
  H-ARCH-REM  : Residual Echo Mixing (linear attention + erased-content echo buffer)
  H-QUANT-SRP : Spectral Residual Packing (PTQ with top-r SVD residual)
  H-OPT-SSM   : Salient-Subspace Muon (orthogonalize only top-k momentum subspace)

Anti-simulation: every metric comes from executed code; ledger is append-only.
"""
from __future__ import annotations

import argparse
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
LEDGER.parent.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_hash() -> str:
    data = Path(__file__).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def append_ledger(entry: Dict[str, Any]) -> None:
    entry = dict(entry)
    entry.setdefault("logged_utc", utc_now())
    entry.setdefault("code_hash", code_hash())
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_block() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threads": torch.get_num_threads(),
        "hostname": platform.node(),
    }


# ---------------------------------------------------------------------------
# H-ARCH-REM: Residual Echo Mixing vs baselines on MQAR
# ---------------------------------------------------------------------------


class SoftmaxAttentionMixer(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class GatedDeltaMixer(nn.Module):
    """Simplified Gated-Delta-like recurrent mixer (tied erase/write scalar β)."""

    def __init__(self, d: int, expand: int = 1):
        super().__init__()
        self.d = d
        self.ed = d * expand
        self.in_proj = nn.Linear(d, 4 * self.ed, bias=False)  # k,v,beta,decay logits
        self.out_proj = nn.Linear(self.ed, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        kvbd = self.in_proj(x)
        k, v, beta_l, decay_l = kvbd.split(self.ed, dim=-1)
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(beta_l)
        alpha = torch.sigmoid(decay_l)
        S = x.new_zeros(B, self.ed, self.ed)
        outs = []
        for t in range(T):
            kt = k[:, t]  # [B, E]
            vt = v[:, t]
            bt = beta[:, t]
            at = alpha[:, t]
            # decay
            S = S * at.unsqueeze(-1)
            # read then erase-write with tied gate (Gated DeltaNet-style)
            read = torch.einsum("bij,bj->bi", S, kt)
            erased = bt * read
            S = S - torch.einsum("bi,bj->bij", erased, kt)
            S = S + torch.einsum("bi,bj->bij", bt * vt, kt)
            y = torch.einsum("bij,bj->bi", S, kt)
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return self.out_proj(y)


class ResidualEchoMixer(nn.Module):
    """
    Residual Echo Mixing (REM) — candidate discovery.

    Novel ingredient vs Gated DeltaNet / GDN-2 family:
      Maintain a second low-rank echo state E_t that accumulates the *erased*
      content through a learned projector P, with slower decay γ.
      Output mixes recurrent read with echo readout.

    Update (per timestep, batch omitted):
      α_t = σ(·),  b_t = σ(·),  w_t = σ(·)          # channel-wise
      S ← α_t ⊙ S
      read ← S k̂
      erase ← b_t ⊙ read
      S ← S − erase k̂ᵀ + (w_t ⊙ v) k̂ᵀ
      e ← P(erase)                                  # low-rank echo write
      E ← γ ⊙ E + (1−γ) ⊙ e
      y ← S k̂ + β_echo * E

    Delta vs nearest neighbors:
      - Gated DeltaNet / KDA / GDN-2: irreversible erase; no echo buffer
      - Standard SSM: no delta erase-before-write + echo residual path
    """

    def __init__(self, d: int, echo_dim: Optional[int] = None, expand: int = 1):
        super().__init__()
        self.d = d
        self.ed = d * expand
        self.echo_dim = echo_dim or max(8, d // 4)
        # k, v, erase, write, decay, echo_decay
        self.in_proj = nn.Linear(d, 5 * self.ed + self.echo_dim, bias=False)
        self.erase_to_echo = nn.Linear(self.ed, self.echo_dim, bias=False)
        self.echo_scale = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(self.ed + self.echo_dim, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        parts = self.in_proj(x)
        k, v, erase_l, write_l, decay_l, echo_decay_l = torch.split(
            parts,
            [self.ed, self.ed, self.ed, self.ed, self.ed, self.echo_dim],
            dim=-1,
        )
        k = F.normalize(k, dim=-1)
        b = torch.sigmoid(erase_l)
        w = torch.sigmoid(write_l)
        alpha = torch.sigmoid(decay_l)
        gamma = torch.sigmoid(echo_decay_l)
        S = x.new_zeros(B, self.ed, self.ed)
        E = x.new_zeros(B, self.echo_dim)
        outs = []
        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]
            bt = b[:, t]
            wt = w[:, t]
            at = alpha[:, t]
            gt = gamma[:, t]
            S = S * at.unsqueeze(-1)
            read = torch.einsum("bij,bj->bi", S, kt)
            erase = bt * read
            S = S - torch.einsum("bi,bj->bij", erase, kt)
            S = S + torch.einsum("bi,bj->bij", wt * vt, kt)
            echo_write = self.erase_to_echo(erase)
            E = gt * E + (1.0 - gt) * echo_write
            y_s = torch.einsum("bij,bj->bi", S, kt)
            y = torch.cat([y_s, self.echo_scale * E], dim=-1)
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return self.out_proj(y)


class MixerLM(nn.Module):
    def __init__(self, vocab: int, d: int, mixer: str, n_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layers = []
        for _ in range(n_layers):
            if mixer == "attn":
                m = SoftmaxAttentionMixer(d)
            elif mixer == "gated_delta":
                m = GatedDeltaMixer(d)
            elif mixer == "rem":
                m = ResidualEchoMixer(d)
            else:
                raise ValueError(mixer)
            layers.append(nn.ModuleDict({"norm": nn.LayerNorm(d), "mix": m, "ffn_n": nn.LayerNorm(d), "ffn": nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
            )}))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        for layer in self.layers:
            h = h + layer["mix"](layer["norm"](h))
            h = h + layer["ffn"](layer["ffn_n"](h))
        return self.head(h)


def make_mqar_batch(
    batch: int,
    n_pairs: int,
    n_queries: int,
    vocab: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-Query Associative Recall.
    Sequence: [k1,v1,...,kN,vN, q1,...,qM] ; labels at query positions = bound values.
    Keys/values drawn from disjoint vocab ranges.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    key_vocab = vocab // 2
    # reserve 0 as pad
    keys = torch.randint(1, key_vocab, (batch, n_pairs), generator=g)
    vals = torch.randint(key_vocab, vocab, (batch, n_pairs), generator=g)
    # queries sample from keys (with replacement)
    q_idx = torch.randint(0, n_pairs, (batch, n_queries), generator=g)
    queries = torch.gather(keys, 1, q_idx)
    targets = torch.gather(vals, 1, q_idx)
    seq_list = []
    for b in range(batch):
        toks = []
        for i in range(n_pairs):
            toks.extend([keys[b, i].item(), vals[b, i].item()])
        toks.extend(queries[b].tolist())
        seq_list.append(toks)
    x = torch.tensor(seq_list, dtype=torch.long, device=device)
    # label mask: only query positions (last n_queries)
    T = x.size(1)
    y = torch.full((batch, T), -100, dtype=torch.long, device=device)
    y[:, -n_queries:] = targets.to(device)
    return x, y, targets.to(device)


@dataclass
class ArchPrereg:
    hyp_id: str = "H-ARCH-REM"
    metric: str = "mqar_accuracy"
    pass_margin: float = 0.05  # absolute acc over gated_delta mean
    min_seeds: int = 3
    effect_vs_noise: str = "mean(candidate)-mean(baseline) > 2 * pooled_std"
    seeds: Tuple[int, ...] = (0, 1, 2)
    steps: int = 250
    batch: int = 32
    n_pairs: int = 12
    n_queries: int = 4
    vocab: int = 64
    d_model: int = 48
    n_layers: int = 2
    lr: float = 3e-3


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_mqar(mixer: str, seed: int, cfg: ArchPrereg, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    model = MixerLM(cfg.vocab, cfg.d_model, mixer, cfg.n_layers).to(device)
    # parameter-match note: REM has echo params; we report param counts honestly
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    t0 = time.time()
    losses = []
    for step in range(cfg.steps):
        x, y, _ = make_mqar_batch(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.vocab, seed * 100000 + step, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for e in range(20):
            x, y, targets = make_mqar_batch(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.vocab, 10_000_000 + seed * 1000 + e, device)
            logits = model(x)
            pred = logits[:, -cfg.n_queries:, :].argmax(dim=-1)
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
    acc = correct / max(total, 1)
    return {
        "mixer": mixer,
        "seed": seed,
        "accuracy": acc,
        "final_loss": losses[-1],
        "mean_last50_loss": sum(losses[-50:]) / len(losses[-50:]),
        "params": count_params(model),
        "wall_clock_s": time.time() - t0,
        "steps": cfg.steps,
    }


def run_arch_l1(device: torch.device) -> Dict[str, Any]:
    cfg = ArchPrereg()
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": cfg.hyp_id,
        "config": asdict(cfg),
        "prediction": "REM mean MQAR acc > gated_delta mean by >= 0.05 and effect > 2σ seed noise",
        "kill_criterion": "REM fails to beat gated_delta by margin OR ablation without echo matches REM",
        "env": env_block(),
    })
    mixers = ["attn", "gated_delta", "rem"]
    all_runs = []
    for mixer in mixers:
        for seed in cfg.seeds:
            try:
                r = train_mqar(mixer, seed, cfg, device)
                r["status"] = "ok"
            except Exception as e:
                r = {"mixer": mixer, "seed": seed, "status": "error", "error": str(e), "trace": traceback.format_exc()}
            all_runs.append(r)
            append_ledger({"type": "result", "gate": "L1", "hyp_id": cfg.hyp_id, "run": r})
            print(json.dumps(r))
    # summarize
    summary = {}
    for mixer in mixers:
        xs = [r["accuracy"] for r in all_runs if r.get("status") == "ok" and r["mixer"] == mixer]
        if xs:
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1)
            summary[mixer] = {"mean_acc": mean, "std_acc": math.sqrt(var), "n": len(xs), "accs": xs,
                              "params": next(r["params"] for r in all_runs if r.get("mixer") == mixer and r.get("status") == "ok")}
    # ablation: REM with echo_scale fixed to 0 via zeroing echo path — kill check
    abl_runs = []
    for seed in cfg.seeds:
        set_seed(seed)
        model = MixerLM(cfg.vocab, cfg.d_model, "rem", cfg.n_layers).to(device)
        # freeze echo contribution
        for layer in model.layers:
            layer["mix"].echo_scale.data.zero_()
            layer["mix"].echo_scale.requires_grad_(False)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0.01)
        for step in range(cfg.steps):
            x, y, _ = make_mqar_batch(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.vocab, seed * 100000 + step, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for e in range(20):
                x, y, targets = make_mqar_batch(cfg.batch, cfg.n_pairs, cfg.n_queries, cfg.vocab, 10_000_000 + seed * 1000 + e, device)
                pred = model(x)[:, -cfg.n_queries:, :].argmax(dim=-1)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())
        abl = {"mixer": "rem_no_echo", "seed": seed, "accuracy": correct / max(total, 1), "params": count_params(model)}
        abl_runs.append(abl)
        append_ledger({"type": "result", "gate": "L1", "hyp_id": cfg.hyp_id, "run": abl, "ablation": "echo_scale=0"})
        print(json.dumps(abl))
    abl_accs = [r["accuracy"] for r in abl_runs]
    abl_mean = sum(abl_accs) / len(abl_accs)
    rem = summary.get("rem")
    base = summary.get("gated_delta")
    verdict = "FAIL"
    reason = ""
    if rem and base:
        delta = rem["mean_acc"] - base["mean_acc"]
        pooled = math.sqrt(0.5 * (rem["std_acc"] ** 2 + base["std_acc"] ** 2))
        echo_delta = rem["mean_acc"] - abl_mean
        if delta >= cfg.pass_margin and delta > 2 * pooled + 1e-12 and echo_delta >= 0.03:
            verdict = "PASS_L1"
            reason = f"delta={delta:.4f} > margin and >2σ ({2*pooled:.4f}); echo ablation gap={echo_delta:.4f}"
        else:
            verdict = "FAIL_L1"
            reason = f"delta={delta:.4f}, 2σ={2*pooled:.4f}, echo_gap={echo_delta:.4f}"
    out = {
        "hyp_id": cfg.hyp_id,
        "summary": summary,
        "ablation_rem_no_echo_mean": abl_mean,
        "ablation_accs": abl_accs,
        "verdict": verdict,
        "reason": reason,
        "runs": all_runs,
    }
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": cfg.hyp_id, "verdict": verdict, "reason": reason, "summary": summary})
    (RESULTS / "arch_rem_l1.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# H-QUANT-SRP: Spectral Residual Packing
# ---------------------------------------------------------------------------


def fake_quantize_symmetric(w: torch.Tensor, bits: int, axis: str = "row") -> torch.Tensor:
    """Symmetric uniform quant. axis='row' => per-output-channel (LLM Linear default)."""
    if bits >= 16:
        return w.clone()
    qmax = 2 ** (bits - 1) - 1
    if w.ndim == 1:
        scale = w.abs().amax().clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
        return q * scale
    if axis == "row":
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    elif axis == "col":
        scale = w.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / qmax
    elif axis == "tensor":
        scale = w.abs().amax().clamp_min(1e-8) / qmax
    else:
        raise ValueError(axis)
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale


def spectral_residual_pack(w: torch.Tensor, base_bits: int, rank: int, residual_bits: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Q = quant_n(W); R = W - Q; R ≈ U_r Σ_r V_r^T quantized.
    Reconstruct Ŵ = Q + Uq Σq Vq^T
    """
    q = fake_quantize_symmetric(w, base_bits, axis="row")
    r = w - q
    # economy SVD
    try:
        U, S, Vh = torch.linalg.svd(r, full_matrices=False)
    except Exception:
        U, S, Vh = torch.svd(r)
        Vh = Vh.transpose(-2, -1)
    rnk = min(rank, S.numel())
    U = U[:, :rnk]
    S = S[:rnk]
    Vh = Vh[:rnk, :]
    # factor matrices: quantize with tensor-scale (small matrices)
    Uq = fake_quantize_symmetric(U, residual_bits, axis="tensor")
    Vq = fake_quantize_symmetric(Vh, residual_bits, axis="tensor")
    # keep S in fp16-equivalent (store as float here; count bits separately)
    recon = q + (Uq * S.unsqueeze(0)) @ Vq
    meta = {
        "rank": float(rnk),
        "base_bits": float(base_bits),
        "residual_bits": float(residual_bits),
    }
    return recon, meta


def effective_bits_srp(rows: int, cols: int, base_bits: int, rank: int, residual_bits: int) -> float:
    """Average bits per weight including residual factors and fp16 singular values."""
    base = rows * cols * base_bits
    # U: rows x r, V: r x cols, S: r * 16
    residual = rows * rank * residual_bits + rank * cols * residual_bits + rank * 16
    return (base + residual) / (rows * cols)


def gptq_like_column(w: torch.Tensor, bits: int, damp: float = 0.01) -> torch.Tensor:
    """
    OBQ/GPTQ-style sequential column quantization with Hessian surrogate H = X^T X
    estimated from random calibration (built by caller via damp on diag of I as weak
    baseline) — here we use H = I + damp* I which reduces to column-wise RTN with
    NO error feedback (honest weak GPTQ-null). Stronger: use empirical H from calib X.
    This function expects optional attribute via closure; we implement calib-H version.
    """
    raise RuntimeError("use gptq_like_with_H")


def gptq_like_with_H(w: torch.Tensor, bits: int, H: torch.Tensor, blocksize: int = 32) -> torch.Tensor:
    """GPTQ-style (Frantar et al.) column quantization with provided Hessian."""
    W = w.clone().float()
    columns = W.size(1)
    hat = torch.zeros_like(W)
    H = H.float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    # damp
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=W.device)
    H[diag, diag] += damp
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
    except Exception:
        Hinv = torch.linalg.pinv(H)
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(count):
            w_j = W1[:, j]
            d = Hinv1[j, j].clamp_min(1e-8)
            # per-channel (column) scale for this column vector
            q_j = fake_quantize_symmetric(w_j, bits)
            Q1[:, j] = q_j
            err = (w_j - q_j) / d
            Err1[:, j] = err
            if j + 1 < count:
                W1[:, j + 1 :] -= err.unsqueeze(1) * Hinv1[j, j + 1 :].unsqueeze(0)
        hat[:, i1:i2] = Q1
        if i2 < columns:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return hat


@dataclass
class QuantPrereg:
    hyp_id: str = "H-QUANT-SRP"
    metric: str = "reconstruction_mse_and_proxy_ce"
    pass_rule: str = "SRP lower MSE than RTN at matched effective bits by >=10% relative, mean over >=3 seeds; AND lower CE on linear probe"
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    rows: int = 256
    cols: int = 256
    base_bits: int = 3
    rank: int = 8
    residual_bits: int = 4


def run_quant_l1(device: torch.device) -> Dict[str, Any]:
    cfg = QuantPrereg()
    eff = effective_bits_srp(cfg.rows, cfg.cols, cfg.base_bits, cfg.rank, cfg.residual_bits)
    bits_floor = max(2, int(math.floor(eff)))
    bits_ceil = max(bits_floor + 1, int(math.ceil(eff)))
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": cfg.hyp_id,
        "config": asdict(cfg),
        "effective_bits_srp": eff,
        "bits_floor": bits_floor,
        "bits_ceil": bits_ceil,
        "prediction": (
            "At matched bit budget: mean MSE(SRP) <= 0.9*mean MSE(RTN_floor) AND "
            "mean MSE(SRP) <= mean MSE(RTN_ceil) AND mean KL(SRP) < mean KL(RTN_floor). "
            "Secondary: MSE(SRP) <= MSE(GPTQ_ceil) OR KL(SRP) < KL(GPTQ_ceil)."
        ),
        "kill_criterion": "Fails primary MSE/KL gates vs RTN at floor/ceil bit budgets",
        "env": env_block(),
        "note": "Re-run after fixing column-scale degeneracy in GPTQ-lite baseline",
    })
    runs = []
    for seed in cfg.seeds:
        set_seed(seed)
        # synthetic LLM-like weight: low-rank + sparse outliers
        U = torch.randn(cfg.rows, 32, device=device) / math.sqrt(32)
        V = torch.randn(32, cfg.cols, device=device) / math.sqrt(32)
        W = U @ V
        outlier = torch.zeros_like(W)
        idx = torch.randperm(cfg.rows)[: max(1, cfg.rows // 20)]
        outlier[idx] = 8.0 * torch.randn(len(idx), cfg.cols, device=device)
        W = W + 0.05 * torch.randn_like(W) + outlier

        # calibration activations for GPTQ Hessian and KL proxy
        X = torch.randn(512, cfg.cols, device=device)
        Y = X @ W.T
        H = (X.T @ X) / X.size(0)

        srp, meta = spectral_residual_pack(W, cfg.base_bits, cfg.rank, cfg.residual_bits)
        rtn_base = fake_quantize_symmetric(W, cfg.base_bits, axis="row")
        rtn_floor = fake_quantize_symmetric(W, bits_floor, axis="row")
        rtn_ceil = fake_quantize_symmetric(W, bits_ceil, axis="row")
        gptq_ceil = gptq_like_with_H(W, bits_ceil, H)
        gptq_floor = gptq_like_with_H(W, bits_floor, H)

        def metrics(What: torch.Tensor) -> Dict[str, float]:
            mse = float(torch.mean((What - W) ** 2).item())
            logits_fp = Y
            logits_q = X @ What.T
            ce = float(F.kl_div(F.log_softmax(logits_q, dim=-1), F.softmax(logits_fp, dim=-1), reduction="batchmean").item())
            rel_err = float((torch.norm(What - W) / torch.norm(W).clamp_min(1e-8)).item())
            return {"mse": mse, "kl": ce, "rel_fro": rel_err}

        row = {
            "seed": seed,
            "eff_bits_srp": eff,
            "bits_floor": bits_floor,
            "bits_ceil": bits_ceil,
            "srp": metrics(srp),
            "rtn_base_bits": metrics(rtn_base),
            "rtn_floor": metrics(rtn_floor),
            "rtn_ceil": metrics(rtn_ceil),
            "gptq_floor": metrics(gptq_floor),
            "gptq_ceil": metrics(gptq_ceil),
        }
        runs.append(row)
        append_ledger({"type": "result", "gate": "L1", "hyp_id": cfg.hyp_id, "run": row})
        print(json.dumps(row))

    def mean_key(method: str, key: str) -> float:
        return sum(r[method][key] for r in runs) / len(runs)

    means = {
        "srp_mse": mean_key("srp", "mse"),
        "rtn_floor_mse": mean_key("rtn_floor", "mse"),
        "rtn_ceil_mse": mean_key("rtn_ceil", "mse"),
        "gptq_floor_mse": mean_key("gptq_floor", "mse"),
        "gptq_ceil_mse": mean_key("gptq_ceil", "mse"),
        "srp_kl": mean_key("srp", "kl"),
        "rtn_floor_kl": mean_key("rtn_floor", "kl"),
        "rtn_ceil_kl": mean_key("rtn_ceil", "kl"),
        "gptq_ceil_kl": mean_key("gptq_ceil", "kl"),
    }
    primary = (
        means["srp_mse"] <= 0.9 * means["rtn_floor_mse"]
        and means["srp_mse"] <= means["rtn_ceil_mse"]
        and means["srp_kl"] < means["rtn_floor_kl"]
    )
    secondary = means["srp_mse"] <= means["gptq_ceil_mse"] or means["srp_kl"] < means["gptq_ceil_kl"]
    verdict = "PASS_L1" if (primary and secondary) else ("PASS_L1_PRIMARY_ONLY" if primary else "FAIL_L1")
    reason = json.dumps({"means": means, "primary": primary, "secondary": secondary})
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": cfg.hyp_id, "verdict": verdict, "reason": reason, "means": means})
    out = {
        "hyp_id": cfg.hyp_id,
        "effective_bits": eff,
        "bits_floor": bits_floor,
        "bits_ceil": bits_ceil,
        "runs": runs,
        "verdict": verdict,
        "reason": reason,
        "means": means,
    }
    (RESULTS / "quant_srp_l1.json").write_text(json.dumps(out, indent=2))

    # --- Mutation H-QUANT-SRP2: spectral residual ON TOP of GPTQ_floor ---
    # Compare GPTQ_n + SRP residual vs GPTQ_{n+1} at matched total average bits.
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": "H-QUANT-SRP2",
        "prediction": "MSE(GPTQ_floor+SRP) < MSE(GPTQ_ceil) and KL lower, at <= ceil effective bits",
        "kill_criterion": "composite does not beat GPTQ_ceil on MSE and KL (mean over seeds)",
        "config": {"base": "gptq_floor", "rank": cfg.rank, "residual_bits": cfg.residual_bits},
        "env": env_block(),
    })
    runs2 = []
    for seed in cfg.seeds:
        set_seed(seed)
        U = torch.randn(cfg.rows, 32, device=device) / math.sqrt(32)
        V = torch.randn(32, cfg.cols, device=device) / math.sqrt(32)
        W = U @ V
        outlier = torch.zeros_like(W)
        idx = torch.randperm(cfg.rows)[: max(1, cfg.rows // 20)]
        outlier[idx] = 8.0 * torch.randn(len(idx), cfg.cols, device=device)
        W = W + 0.05 * torch.randn_like(W) + outlier
        X = torch.randn(512, cfg.cols, device=device)
        Y = X @ W.T
        H = (X.T @ X) / X.size(0)
        base_q = gptq_like_with_H(W, bits_floor, H)
        # pack residual of GPTQ error (not RTN)
        r = W - base_q
        try:
            Uu, S, Vh = torch.linalg.svd(r, full_matrices=False)
        except Exception:
            Uu, S, Vh = torch.svd(r)
            Vh = Vh.T
        rnk = min(cfg.rank, S.numel())
        Uu, S, Vh = Uu[:, :rnk], S[:rnk], Vh[:rnk, :]
        Uq = fake_quantize_symmetric(Uu, cfg.residual_bits, axis="tensor")
        Vq = fake_quantize_symmetric(Vh, cfg.residual_bits, axis="tensor")
        srp2 = base_q + (Uq * S.unsqueeze(0)) @ Vq
        gptq_c = gptq_like_with_H(W, bits_ceil, H)
        rtn_c = fake_quantize_symmetric(W, bits_ceil, axis="row")
        eff2 = effective_bits_srp(cfg.rows, cfg.cols, bits_floor, cfg.rank, cfg.residual_bits)

        def metrics(What: torch.Tensor) -> Dict[str, float]:
            mse = float(torch.mean((What - W) ** 2).item())
            ce = float(F.kl_div(F.log_softmax(X @ What.T, dim=-1), F.softmax(Y, dim=-1), reduction="batchmean").item())
            return {"mse": mse, "kl": ce}

        row = {
            "seed": seed,
            "eff_bits": eff2,
            "srp2": metrics(srp2),
            "gptq_ceil": metrics(gptq_c),
            "gptq_floor": metrics(base_q),
            "rtn_ceil": metrics(rtn_c),
        }
        runs2.append(row)
        append_ledger({"type": "result", "gate": "L1", "hyp_id": "H-QUANT-SRP2", "run": row})
        print("SRP2", json.dumps(row))

    m2 = {
        "srp2_mse": sum(r["srp2"]["mse"] for r in runs2) / len(runs2),
        "gptq_ceil_mse": sum(r["gptq_ceil"]["mse"] for r in runs2) / len(runs2),
        "srp2_kl": sum(r["srp2"]["kl"] for r in runs2) / len(runs2),
        "gptq_ceil_kl": sum(r["gptq_ceil"]["kl"] for r in runs2) / len(runs2),
        "gptq_floor_mse": sum(r["gptq_floor"]["mse"] for r in runs2) / len(runs2),
    }
    # also require beating gptq_floor substantially (sanity)
    v2 = "PASS_L1" if (
        m2["srp2_mse"] < m2["gptq_ceil_mse"] and m2["srp2_kl"] < m2["gptq_ceil_kl"] and m2["srp2_mse"] < 0.85 * m2["gptq_floor_mse"]
    ) else "FAIL_L1"
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": "H-QUANT-SRP2", "verdict": v2, "means": m2})
    out2 = {"hyp_id": "H-QUANT-SRP2", "means": m2, "verdict": v2, "runs": runs2, "effective_bits": eff2}
    (RESULTS / "quant_srp2_l1.json").write_text(json.dumps(out2, indent=2))
    out["srp2"] = out2
    return out


# ---------------------------------------------------------------------------
# H-OPT-SSM: Salient-Subspace Muon
# ---------------------------------------------------------------------------


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize matrix via Newton-Schulz (Muon-style)."""
    X = G
    if G.size(0) > G.size(1):
        X = G.T
    # normalize
    X = X / (X.norm() + 1e-7)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class AdamWSimple:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.01):
        self.param_groups = [{"params": list(params), "lr": lr, "betas": betas, "eps": eps, "wd": wd}]
        self.state = {}

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.grad = None

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state.setdefault(p, {})
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["t"] = 0
                st["t"] += 1
                st["m"].mul_(g["betas"][0]).add_(grad, alpha=1 - g["betas"][0])
                st["v"].mul_(g["betas"][1]).addcmul_(grad, grad, value=1 - g["betas"][1])
                mhat = st["m"] / (1 - g["betas"][0] ** st["t"])
                vhat = st["v"] / (1 - g["betas"][1] ** st["t"])
                p.add_(p, alpha=-g["wd"] * g["lr"])
                p.add_(mhat / (vhat.sqrt() + g["eps"]), alpha=-g["lr"])


class MuonFamily:
    """Muon / Salient-Subspace Muon for 2D params; AdamW for others."""

    def __init__(self, named_params, lr=1e-3, mode="muon", k_frac=0.25, ns_steps=5, wd=0.01, adam_lr=None):
        self.lr = lr
        self.adam_lr = adam_lr if adam_lr is not None else lr
        self.mode = mode
        self.k_frac = k_frac
        self.ns_steps = ns_steps
        self.wd = wd
        self.params = []
        self.state = {}
        for name, p in named_params:
            self.params.append((name, p))
            self.state[p] = {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}

    def zero_grad(self):
        for _, p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for name, p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            st = self.state[p]
            st["t"] += 1
            st["m"].mul_(0.95).add_(grad, alpha=0.05)
            # weight decay
            p.mul_(1 - self.lr * self.wd)
            if p.ndim == 2 and min(p.shape) >= 8 and self.mode in ("muon", "ss_muon"):
                M = st["m"]
                # Muon update RMS matched to Adam-ish scale: use 0.2 * sqrt(fan_in) heuristic (Moonlight-style adjust)
                fan_in = p.size(1)
                scale = 0.2 * math.sqrt(max(fan_in, 1))
                if self.mode == "muon":
                    U = newton_schulz(M, self.ns_steps)
                    U = U * scale
                    p.add_(U, alpha=-self.lr)
                else:
                    try:
                        Uu, S, Vh = torch.linalg.svd(M, full_matrices=False)
                    except Exception:
                        Uu, S, Vh = torch.svd(M)
                        Vh = Vh.T
                    rmax = S.numel()
                    k = max(1, int(math.ceil(self.k_frac * rmax)))
                    U_k = Uu[:, :k]
                    S_k = S[:k]
                    V_k = Vh[:k, :]
                    M_sal = (U_k * S_k.unsqueeze(0)) @ V_k
                    M_res = M - M_sal
                    Uo = newton_schulz(M_sal, self.ns_steps)
                    Uo = Uo * scale
                    st["v"].mul_(0.95).addcmul_(M_res, M_res, value=0.05)
                    adam_res = M_res / (st["v"].sqrt() + 1e-8)
                    # mix: salient orthogonal + residual adam (residual already scaled like adam)
                    p.add_(Uo, alpha=-self.lr)
                    p.add_(adam_res, alpha=-self.adam_lr)
            else:
                st["v"].mul_(0.95).addcmul_(grad, grad, value=0.05)
                mhat = st["m"] / (1 - 0.95 ** st["t"])
                vhat = st["v"] / (1 - 0.95 ** st["t"])
                p.add_(mhat / (vhat.sqrt() + 1e-8), alpha=-self.adam_lr)


class TinyLM(nn.Module):
    def __init__(self, vocab=128, d=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "n1": nn.LayerNorm(d),
                "attn": SoftmaxAttentionMixer(d, n_heads=4),
                "n2": nn.LayerNorm(d),
                "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }) for _ in range(layers)
        ])
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = h + b["attn"](b["n1"](h))
            h = h + b["mlp"](b["n2"](h))
        return self.head(h)


@dataclass
class OptPrereg:
    hyp_id: str = "H-OPT-SSM"
    metric: str = "val_nll_at_fixed_steps"
    pass_margin: float = 0.02  # absolute NLL improvement vs AdamW and vs Muon
    seeds: Tuple[int, ...] = (0, 1, 2)
    steps: int = 300
    batch: int = 32
    seq: int = 64
    vocab: int = 128
    d: int = 64
    lr: float = 2e-3


def make_toy_text(batch, seq, vocab, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # structured language: repeating motifs + noise (more learnable than pure random)
    base = torch.randint(1, vocab // 4, (batch, seq // 4), generator=g)
    x = base.repeat(1, 4)[:, :seq].contiguous()
    noise = torch.randint(0, 3, (batch, seq), generator=g)
    x = (x + noise) % vocab
    return x.to(device)


def train_opt(mode: str, seed: int, cfg: OptPrereg, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    model = TinyLM(cfg.vocab, cfg.d, layers=2).to(device)
    if mode == "adamw":
        opt: Any = AdamWSimple(model.parameters(), lr=cfg.lr, wd=0.01)
    else:
        opt = MuonFamily(model.named_parameters(), lr=cfg.lr, mode=mode, k_frac=0.25, wd=0.01)
    t0 = time.time()
    last = 0.0
    for step in range(cfg.steps):
        x = make_toy_text(cfg.batch, cfg.seq, cfg.vocab, seed * 10000 + step, device)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    # val
    model.eval()
    vals = []
    with torch.no_grad():
        for e in range(10):
            x = make_toy_text(cfg.batch, cfg.seq, cfg.vocab, 9_000_000 + seed * 100 + e, device)
            inp, tgt = x[:, :-1], x[:, 1:]
            loss = F.cross_entropy(model(inp).reshape(-1, cfg.vocab), tgt.reshape(-1))
            vals.append(float(loss.item()))
    return {
        "mode": mode,
        "seed": seed,
        "val_nll": sum(vals) / len(vals),
        "train_last": last,
        "params": count_params(model),
        "wall_clock_s": time.time() - t0,
    }


def run_opt_l1(device: torch.device) -> Dict[str, Any]:
    cfg = OptPrereg()
    lr_grid = [5e-4, 1e-3, 2e-3, 5e-3]
    append_ledger({
        "type": "preregister",
        "gate": "L1",
        "hyp_id": cfg.hyp_id,
        "config": asdict(cfg),
        "lr_grid": lr_grid,
        "prediction": "After per-method LR selection on seed0: ss_muon mean val_nll < adamw - 0.02 AND < muon - 0.01 over seeds",
        "kill_criterion": "ss_muon does not beat both adamw and full muon on mean val NLL after LR tuning",
        "env": env_block(),
    })
    modes = ["adamw", "muon", "ss_muon"]
    # LR selection on seed 0 only
    best_lr = {}
    for mode in modes:
        scores = []
        for lr in lr_grid:
            cfg_lr = OptPrereg(lr=lr)
            r = train_opt(mode, seed=0, cfg=cfg_lr, device=device)
            scores.append((r["val_nll"], lr, r))
            append_ledger({"type": "lr_sweep", "gate": "L1", "hyp_id": cfg.hyp_id, "mode": mode, "lr": lr, "run": r})
            print("LR", mode, lr, r["val_nll"])
        scores.sort()
        best_lr[mode] = scores[0][1]
    append_ledger({"type": "lr_selected", "hyp_id": cfg.hyp_id, "best_lr": best_lr})

    runs = []
    for mode in modes:
        cfg_m = OptPrereg(lr=best_lr[mode])
        for seed in cfg.seeds:
            r = train_opt(mode, seed, cfg_m, device)
            r["lr"] = best_lr[mode]
            runs.append(r)
            append_ledger({"type": "result", "gate": "L1", "hyp_id": cfg.hyp_id, "run": r})
            print(json.dumps(r))
    summary = {}
    for mode in modes:
        xs = [r["val_nll"] for r in runs if r["mode"] == mode]
        mean = sum(xs) / len(xs)
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1))
        summary[mode] = {"mean_nll": mean, "std": std, "nlls": xs, "lr": best_lr[mode]}
    d_adam = summary["adamw"]["mean_nll"] - summary["ss_muon"]["mean_nll"]
    d_muon = summary["muon"]["mean_nll"] - summary["ss_muon"]["mean_nll"]
    verdict = "PASS_L1" if (d_adam >= cfg.pass_margin and d_muon >= 0.01) else "FAIL_L1"
    reason = f"d_adam={d_adam:.4f} d_muon={d_muon:.4f} summary={summary}"
    append_ledger({"type": "verdict", "gate": "L1", "hyp_id": cfg.hyp_id, "verdict": verdict, "reason": reason, "summary": summary})
    out = {"hyp_id": cfg.hyp_id, "summary": summary, "verdict": verdict, "reason": reason, "runs": runs, "best_lr": best_lr}
    (RESULTS / "opt_ssm_l1.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# L0 screen for additional hypotheses (no compute beyond bookkeeping)
# ---------------------------------------------------------------------------

L0_CANDIDATES = [
    {"id": "H-ARCH-REM", "domain": "architecture", "novel": True, "note": "echo buffer of erased delta-rule content", "decision": "PROMOTE_L1"},
    {"id": "H-QUANT-SRP", "domain": "quantization", "novel": True, "note": "spectral residual packing at matched bit budget", "decision": "PROMOTE_L1"},
    {"id": "H-OPT-SSM", "domain": "training", "novel": True, "note": "salient-subspace Muon", "decision": "PROMOTE_L1"},
    {"id": "H-ARCH-GATEBIAS", "domain": "architecture", "novel": False, "note": "gate bias half-life = hyperparam of GDN", "decision": "KILL_L0_reducible"},
    {"id": "H-QNT-ROTSIDE", "domain": "quantization", "novel": False, "note": "rotation side choice explored in SpinQuant/QuaRot family", "decision": "KILL_L0_reducible"},
    {"id": "H-OPT-NSSTEPS", "domain": "training", "novel": False, "note": "Newton-Schulz step count tuning", "decision": "KILL_L0_reducible"},
    {"id": "H-ARCH-CONFLICT-ERASE", "domain": "architecture", "novel": True, "note": "modulate erase by key-collision energy", "decision": "HOLD_queue"},
    {"id": "H-TRAIN-SVA-STUDENT-SUPPORT", "domain": "training", "novel": True, "note": "student-selected salient support for OPD", "decision": "HOLD_needs_teacher_infra"},
    {"id": "H-QNT-SIGN-RESIDUAL", "domain": "quantization", "novel": True, "note": "sign-consistent sparse residual codebook", "decision": "HOLD_queue"},
    {"id": "H-ARCH-ASYM-STATE-RANK", "domain": "architecture", "novel": True, "note": "asymmetric key/value state ranks in delta rule", "decision": "HOLD_queue"},
]


def run_l0() -> Dict[str, Any]:
    for c in L0_CANDIDATES:
        append_ledger({"type": "l0_screen", "gate": "L0", **c})
    promoted = [c for c in L0_CANDIDATES if c["decision"] == "PROMOTE_L1"]
    killed = [c for c in L0_CANDIDATES if c["decision"].startswith("KILL")]
    held = [c for c in L0_CANDIDATES if c["decision"].startswith("HOLD")]
    out = {"promoted": promoted, "killed": killed, "held": held, "n": len(L0_CANDIDATES)}
    (RESULTS / "l0_screen.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "l0", "arch", "quant", "opt"], default="all")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    append_ledger({"type": "session_start", "env": env_block(), "only": args.only})
    results = {}
    if args.only in ("all", "l0"):
        results["l0"] = run_l0()
    if args.only in ("all", "arch"):
        results["arch"] = run_arch_l1(device)
    if args.only in ("all", "quant"):
        results["quant"] = run_quant_l1(device)
    if args.only in ("all", "opt"):
        results["opt"] = run_opt_l1(device)
    (RESULTS / "l1_summary.json").write_text(json.dumps(results, indent=2, default=str))
    append_ledger({"type": "session_end", "summary_keys": list(results.keys())})
    print("=== SUMMARY ===")
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else v) for k, v in results.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
