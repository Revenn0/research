#!/usr/bin/env python3
"""Benchmark ATLAS-LM 230M (arquitetura LFM2 padrão) vs LiquidAI LFM2.5-230M-Base."""

from __future__ import annotations

import gc
import json
import math
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Lfm2Config, Lfm2ForCausalLM


LIQUID_REPO = "LiquidAI/LFM2.5-230M-Base"
WIKITEXT_REPO = "Salesforce/wikitext"


@dataclass
class BenchmarkConfig:
    seq_len: int = 128
    train_steps: int = 200
    eval_batches: int = 40
    batch_size: int = 1
    lr: float = 3e-4
    seed: int = 42
    device: str = "cpu"
    threads: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_wikitext_split(split: str) -> list[str]:
    files = {
        "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
        "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    }
    path = hf_hub_download(WIKITEXT_REPO, files[split], repo_type="dataset")
    table = pq.read_table(path)
    return [t for t in table.column("text").to_pylist() if t and t.strip()]


def tokenize_corpus(tokenizer, texts: list[str], max_tokens: int | None = None) -> torch.Tensor:
    joined = "\n".join(texts)
    ids = tokenizer(joined, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return ids


def make_batches(data: torch.Tensor, seq_len: int, batch_size: int, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    max_start = len(data) - seq_len - 1
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_batches):
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[s : s + seq_len] for s in starts])
        y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        batches.append((x, y))
    return batches


@torch.no_grad()
def eval_perplexity(
    model,
    data: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
) -> dict:
    model.eval()
    batches = make_batches(data, seq_len, batch_size, n_batches)
    losses = []
    tokens = 0
    t0 = time.time()
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        out = model(x, labels=y)
        losses.append(out.loss.item())
        tokens += y.numel()
    wall = time.time() - t0
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "tokens_evaluated": tokens,
        "eval_wall_s": wall,
        "eval_tok_per_s": tokens / max(wall, 1e-9),
    }


def benchmark_forward(model, seq_len: int, device: torch.device, n_iters: int = 5) -> dict:
    model.eval()
    x = torch.randint(1, 5000, (1, seq_len), device=device)
    # warmup
    with torch.no_grad():
        model(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            model(x)
    wall = time.time() - t0
    tokens = n_iters * seq_len
    return {
        "forward_iters": n_iters,
        "forward_wall_s": wall,
        "forward_tok_per_s": tokens / max(wall, 1e-9),
    }


def train_from_scratch(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    cfg: BenchmarkConfig,
    name: str,
) -> dict:
    device = torch.device(cfg.device)
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    train_losses: list[float] = []
    for step in range(1, cfg.train_steps + 1):
        max_start = len(train_data) - cfg.seq_len - 1
        starts = torch.randint(0, max_start, (cfg.batch_size,))
        x = torch.stack([train_data[s : s + cfg.seq_len] for s in starts]).to(device)
        y = torch.stack([train_data[s + 1 : s + cfg.seq_len + 1] for s in starts]).to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_losses.append(loss.item())
        if step % max(1, cfg.train_steps // 4) == 0:
            print(f"  [{name}] step {step}/{cfg.train_steps} loss={loss.item():.4f}")
    wall = time.time() - t0
    val_metrics = eval_perplexity(
        model,
        val_data,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        n_batches=cfg.eval_batches,
        device=device,
    )
    return {
        "name": name,
        "params_m": n_params / 1e6,
        "train_steps": cfg.train_steps,
        "train_wall_s": wall,
        "train_steps_per_s": cfg.train_steps / max(wall, 1e-9),
        "final_train_loss": train_losses[-1],
        "mean_train_loss": sum(train_losses) / len(train_losses),
        **val_metrics,
        "peak_rss_mb": peak_rss_mb(),
    }


def build_pure_attention_config(base: Lfm2Config) -> Lfm2Config:
    """Mesmo tamanho, mas 100% full_attention (ablation 'normal transformer')."""
    cfg = Lfm2Config(**base.to_dict())
    cfg.layer_types = ["full_attention"] * base.num_hidden_layers
    return cfg


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    device = torch.device(cfg.device)

    print("Carregando tokenizer e WikiText-2...")
    tokenizer = AutoTokenizer.from_pretrained(LIQUID_REPO, trust_remote_code=True)
    train_texts = load_wikitext_split("train")
    val_texts = load_wikitext_split("validation")
    # Limita tokens para caber na VM e terminar em tempo razoável.
    train_ids = tokenize_corpus(tokenizer, train_texts, max_tokens=200_000)
    val_ids = tokenize_corpus(tokenizer, val_texts, max_tokens=80_000)
    print(f"  train tokens: {len(train_ids):,} | val tokens: {len(val_ids):,}")

    liquid_cfg = Lfm2Config.from_pretrained(LIQUID_REPO)
    print(f"  LFM2 config: layers={liquid_cfg.num_hidden_layers}, hidden={liquid_cfg.hidden_size}, layer_types={liquid_cfg.layer_types}")

    results: dict = {
        "protocol": {
            "dataset": "WikiText-2-raw-v1",
            "tokenizer": LIQUID_REPO,
            "seq_len": cfg.seq_len,
            "train_steps": cfg.train_steps,
            "eval_batches": cfg.eval_batches,
            "device": cfg.device,
            "note": "Liquid é pré-treinado (28T tokens); ATLAS from-scratch usa mesma arquitetura LFM2.",
        },
        "models": {},
    }

    # --- 1) Liquid pré-treinado (referência) ---
    print("\n=== Liquid LFM2.5-230M-Base (pré-treinado) ===")
    liquid = AutoModelForCausalLM.from_pretrained(LIQUID_REPO, torch_dtype=torch.float32).to(device)
    liquid.eval()
    n_liquid = sum(p.numel() for p in liquid.parameters())
    liquid_eval = eval_perplexity(
        liquid, val_ids, seq_len=cfg.seq_len, batch_size=cfg.batch_size, n_batches=cfg.eval_batches, device=device
    )
    liquid_fwd = benchmark_forward(liquid, cfg.seq_len, device)
    results["models"]["liquid_pretrained"] = {
        "params_m": n_liquid / 1e6,
        "architecture": "LFM2 hybrid (8 conv + 6 full_attention)",
        "pretrained": True,
        **liquid_eval,
        **liquid_fwd,
        "peak_rss_mb": peak_rss_mb(),
    }
    print(
        f"  PPL={liquid_eval['perplexity']:.2f} loss={liquid_eval['loss']:.4f} "
        f"fwd={liquid_fwd['forward_tok_per_s']:.0f} tok/s"
    )
    del liquid
    gc.collect()

    # --- 2) ATLAS-LM 230M normal = LFM2 from scratch (mesma config) ---
    print("\n=== ATLAS-LM 230M normal (LFM2 hybrid, from scratch) ===")
    atlas_hybrid = Lfm2ForCausalLM(liquid_cfg)
    atlas_hybrid_result = train_from_scratch(
        atlas_hybrid, train_ids, val_ids, cfg=cfg, name="atlas_lfm2_hybrid_scratch"
    )
    atlas_hybrid_result["architecture"] = "LFM2 hybrid (idêntico ao Liquid)"
    atlas_hybrid_result["pretrained"] = False
    atlas_hybrid_result.update(benchmark_forward(atlas_hybrid.to(device), cfg.seq_len, device))
    results["models"]["atlas_lfm2_hybrid_scratch"] = atlas_hybrid_result
    print(
        f"  PPL={atlas_hybrid_result['perplexity']:.2f} loss={atlas_hybrid_result['loss']:.4f} "
        f"train={atlas_hybrid_result['train_steps_per_s']:.3f} steps/s"
    )
    del atlas_hybrid
    gc.collect()

    # --- 3) Ablation: 100% attention (mesmo param count ~) ---
    print("\n=== ATLAS-LM 230M ablation (100% full_attention, from scratch) ===")
    pure_cfg = build_pure_attention_config(liquid_cfg)
    atlas_pure = Lfm2ForCausalLM(pure_cfg)
    n_pure = sum(p.numel() for p in atlas_pure.parameters())
    atlas_pure_result = train_from_scratch(
        atlas_pure, train_ids, val_ids, cfg=cfg, name="atlas_lfm2_pure_attn_scratch"
    )
    atlas_pure_result["architecture"] = "LFM2 100% full_attention"
    atlas_pure_result["pretrained"] = False
    atlas_pure_result["params_m"] = n_pure / 1e6
    atlas_pure_result.update(benchmark_forward(atlas_pure.to(device), cfg.seq_len, device))
    results["models"]["atlas_lfm2_pure_attn_scratch"] = atlas_pure_result
    print(
        f"  PPL={atlas_pure_result['perplexity']:.2f} loss={atlas_pure_result['loss']:.4f} "
        f"train={atlas_pure_result['train_steps_per_s']:.3f} steps/s"
    )
    del atlas_pure
    gc.collect()

    # --- Summary ---
    liq = results["models"]["liquid_pretrained"]
    hyb = results["models"]["atlas_lfm2_hybrid_scratch"]
    pure = results["models"]["atlas_lfm2_pure_attn_scratch"]
    results["summary"] = {
        "liquid_pretrained_ppl": liq["perplexity"],
        "atlas_hybrid_scratch_ppl": hyb["perplexity"],
        "atlas_pure_attn_scratch_ppl": pure["perplexity"],
        "hybrid_vs_pure_attn_delta_ppl": hyb["perplexity"] - pure["perplexity"],
        "hybrid_train_speedup_vs_pure_attn": pure["train_wall_s"] / max(hyb["train_wall_s"], 1e-9),
        "hybrid_forward_speedup_vs_pure_attn": pure["forward_tok_per_s"] / max(hyb["forward_tok_per_s"], 1e-9),
        "liquid_forward_tok_per_s": liq["forward_tok_per_s"],
        "atlas_hybrid_forward_tok_per_s": hyb["forward_tok_per_s"],
    }
    return results


def main() -> None:
    cfg = BenchmarkConfig()
    print("Benchmark ATLAS-LM 230M vs Liquid AI 230M")
    print(json.dumps(asdict(cfg), indent=2))
    results = run_benchmark(cfg)
    out = Path("benchmark_liquid_230m_results.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\n=== RESUMO ===")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()
