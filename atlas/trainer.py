"""Shared training loop for ATLAS char-level LM."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from atlas.datasets import load_train_val, random_batch
from atlas.models import CharLM
from atlas.optimizers import build_optimizer


@dataclass
class LMTrainConfig:
    tag: str = "run"
    seed: int = 42
    total_steps: int = 8000
    eval_every: int = 1000
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 64
    block_size: int = 64
    batch: int = 16
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.01
    mixing: str = "global-attn"
    norm: str = "layernorm"
    activation: str = "gelu"
    optimizer: str = "adamw"
    schedule: str = "cosine"
    data_path: str = "atlas/data/tinyshakespeare.txt"
    log_every: int = 200
    grad_clip: float = 1.0
    warmup_steps: int = 200
    device: str = "auto"
    amp: bool = False
    label_smoothing: float = 0.0


@dataclass
class LMTrainResult:
    tag: str
    seed: int
    total_steps: int
    wall_time_s: float
    steps_per_sec: float
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    val_losses: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_at_step(cfg: LMTrainConfig, step: int) -> float:
    if cfg.schedule == "none":
        return cfg.lr
    if cfg.schedule == "cosine":
        progress = step / max(cfg.total_steps, 1)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if cfg.schedule == "cosine-warmup":
        if step < cfg.warmup_steps:
            return cfg.lr * step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown schedule: {cfg.schedule}")


@torch.no_grad()
def evaluate(
    model: CharLM, data: torch.Tensor, block_size: int, batch: int, device: torch.device
) -> float:
    model.eval()
    losses = []
    n_eval = min(20, max(1, (len(data) - block_size - 1) // batch))
    for start in range(0, n_eval * batch, batch):
        idx = torch.arange(start, start + batch)
        x = torch.stack([data[i : i + block_size] for i in idx]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(sum(losses) / len(losses))


def train_lm(cfg: LMTrainConfig) -> LMTrainResult:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    if device.type == "cpu":
        torch.set_num_threads(2)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()

    train_ds, val_ds = load_train_val(cfg.data_path)
    assert train_ds.stoi == val_ds.stoi
    assert train_ds.vocab_size == val_ds.vocab_size

    model = CharLM(
        vocab_size=train_ds.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        mixing=cfg.mixing,
        norm=cfg.norm,
        activation=cfg.activation,
    ).to(device)
    opt = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay)

    val_losses: list[float] = []
    best_val = float("inf")
    train_loss = 0.0

    for step in range(1, cfg.total_steps + 1):
        model.train()
        lr = lr_at_step(cfg, step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = random_batch(train_ds.data, cfg.block_size, cfg.batch)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        train_loss = loss.item()

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            vloss = evaluate(model, val_ds.data, cfg.block_size, cfg.batch, device)
            val_losses.append(vloss)
            best_val = min(best_val, vloss)

    wall = time.time() - t0
    final_val = evaluate(model, val_ds.data, cfg.block_size, cfg.batch, device)
    best_val = min(best_val, final_val)

    return LMTrainResult(
        tag=cfg.tag,
        seed=cfg.seed,
        total_steps=cfg.total_steps,
        wall_time_s=wall,
        steps_per_sec=cfg.total_steps / max(wall, 1e-9),
        final_train_loss=train_loss,
        final_val_loss=final_val,
        best_val_loss=best_val,
        val_losses=val_losses,
        config=asdict(cfg),
    )


def append_experiment_log(entry: dict, log_path: str = "atlas/experiments_log.jsonl") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def result_to_log_entry(result: LMTrainResult, command: str) -> dict:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "tag": result.tag,
        "seed": result.seed,
        "final_val_loss": result.final_val_loss,
        "best_val_loss": result.best_val_loss,
        "wall_time_s": result.wall_time_s,
        "config": result.config,
    }
