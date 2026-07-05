#!/usr/bin/env python3
"""Trainer de referência mínimo ATLAS — patches aplicados via flags, nunca reescrita total."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class TrainConfig:
    seed: int = 42
    steps: int = 800
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.1
    label_smoothing: float = 0.0
    warmup_steps: int = 0
    grad_clip: float = 0.0
    grad_centralize: bool = False
    # patches ATLAS
    optimizer: str = "adam"  # adam | adamw | sgd_momentum
    scheduler: str = "none"  # none | cosine | sqrt
    activation: str = "relu"  # relu | gelu | silu | spatial_gate_relu | signed_sqrt | variance_gated_relu
    norm: str = "none"  # none | batchnorm | layernorm
    mixing: str = "none"  # none | local_blend
    init_scale: float = 1.0
    ema_decay: float = 0.0
    mixup_alpha: float = 0.0
    cutout_size: int = 0
    data_dir: str = "./data"
    num_workers: int = 0
    log_every: int = 200
    eval_every: int = 200


@dataclass
class TrainResult:
    seed: int
    steps: int
    wall_time_s: float
    steps_per_sec: float
    final_train_loss: float
    final_train_acc: float
    best_val_acc: float
    final_val_acc: float
    val_accs: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_activation(name: str) -> type[nn.Module]:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]


def apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "spatial_gate_relu":
        if x.dim() == 4:
            energy = x.abs().mean(dim=(2, 3), keepdim=True)
            ref = energy.mean(dim=1, keepdim=True).clamp(min=1e-5)
            gate = (energy / ref).clamp(0.0, 2.0)
            return F.relu(x) * gate
        return F.relu(x)
    if name == "signed_sqrt":
        return x.sign() * x.abs().sqrt()
    if name == "variance_gated_relu":
        if x.dim() == 4:
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            gate = torch.sigmoid(var * 4.0)
            return F.relu(x) * gate
        return F.relu(x)
    return get_activation(name)()


class LocalBlend(nn.Module):
    """NOVEL: mistura local depthwise com gate escalar por canal (média espacial)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw(x)
        gate = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * x


class SmallCNN(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.activation_name = cfg.activation
        use_builtin_act = cfg.activation in ("relu", "gelu", "silu")
        c1, c2 = 32, 64
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = self._make_norm(cfg.norm, c1)
        self.norm2 = self._make_norm(cfg.norm, c2)
        self.mix1 = LocalBlend(c1) if cfg.mixing == "local_blend" else nn.Identity()
        self.mix2 = LocalBlend(c2) if cfg.mixing == "local_blend" else nn.Identity()
        self.act = get_activation(cfg.activation)() if use_builtin_act else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)
        flat = c2 * 7 * 7
        self.fc1 = nn.Linear(flat, cfg.hidden_dim)
        self.fc_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.norm == "layernorm" else nn.Identity()
        self.fc2 = nn.Linear(cfg.hidden_dim, 10)
        self._init_weights(cfg.init_scale)

    @staticmethod
    def _make_norm(norm: str, channels: int) -> nn.Module:
        if norm == "batchnorm":
            return nn.BatchNorm2d(channels)
        if norm == "layernorm":
            return nn.GroupNorm(1, channels)
        return nn.Identity()

    def _init_weights(self, scale: float) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.weight.data.mul_(scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name in ("relu", "gelu", "silu"):
            return self.act(x)
        return apply_activation(x, self.activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self._activate(self.mix1(self.norm1(self.conv1(x)))))
        x = self.pool(self._activate(self.mix2(self.norm2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self._activate(self.fc_norm(self.fc1(x))))
        return self.fc2(x)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def cutout(x: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    y = torch.randint(0, h, (1,)).item()
    x0 = torch.randint(0, w, (1,)).item()
    y0 = max(0, y - size // 2)
    x0 = max(0, x0 - size // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    x = x.clone()
    x[..., y0:y1, x0:x1] = 0.0
    return x


class NormFeedbackState:
    """NOVEL: feedback global de norma de gradiente (escalar único, não por-parâmetro)."""

    def __init__(self, target_norm: float = 1.0, ema_decay: float = 0.99) -> None:
        self.target_norm = target_norm
        self.ema_decay = ema_decay
        self.running_norm = target_norm

    def scale_gradients(self, model: nn.Module) -> None:
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += p.grad.data.pow(2).sum().item()
        grad_norm = math.sqrt(total_sq) + 1e-8
        self.running_norm = self.ema_decay * self.running_norm + (1 - self.ema_decay) * grad_norm
        scale = self.target_norm / (self.running_norm + 1e-8)
        scale = max(0.25, min(4.0, scale))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_ds = datasets.FashionMNIST(cfg.data_dir, train=True, download=True, transform=tfm)
    val_ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "norm_feedback":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def lr_at_step(cfg: TrainConfig, step: int, total_steps: int) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.scheduler == "cosine":
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
    if cfg.scheduler == "sqrt":
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return cfg.lr / math.sqrt(1 + 9 * progress)
    return cfg.lr


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train(cfg: TrainConfig) -> TrainResult:
    set_seed(cfg.seed)
    device = torch.device("cpu")
    train_loader, val_loader = build_loaders(cfg)
    model = SmallCNN(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    norm_fb = NormFeedbackState() if cfg.optimizer == "norm_feedback" else None
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    train_iter = iter(train_loader)
    step = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    val_accs: list[float] = []
    best_val_acc = 0.0

    t0 = time.perf_counter()
    while step < cfg.steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        model.train()
        x, y = x.to(device), y.to(device)
        if cfg.cutout_size > 0:
            x = cutout(x, cfg.cutout_size)
        y_a, y_b, lam = y, y, 1.0
        if cfg.mixup_alpha > 0:
            x, y_a, y_b, lam = mixup_batch(x, y, cfg.mixup_alpha)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_at_step(cfg, step, cfg.steps)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if cfg.mixup_alpha > 0:
            loss = lam * F.cross_entropy(logits, y_a, label_smoothing=cfg.label_smoothing) + (
                1 - lam
            ) * F.cross_entropy(logits, y_b, label_smoothing=cfg.label_smoothing)
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

        loss.backward()
        if cfg.optimizer == "norm_feedback" and norm_fb is not None:
            norm_fb.scale_gradients(model)
        if cfg.grad_centralize:
            for p in model.parameters():
                if p.grad is not None and p.grad.dim() > 1:
                    p.grad.data -= p.grad.data.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True)
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        if cfg.mixup_alpha > 0:
            running_correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
        else:
            running_correct += (pred == y).sum().item()
        running_total += x.size(0)
        step += 1

        if step % cfg.eval_every == 0 or step == cfg.steps:
            eval_model = model
            if ema is not None:
                eval_model = SmallCNN(cfg).to(device)
                ema.copy_to(eval_model)
            vloss, vacc = evaluate(eval_model, val_loader, device)
            val_accs.append(vacc)
            best_val_acc = max(best_val_acc, vacc)

    wall = time.perf_counter() - t0
    final_train_loss = running_loss / max(1, running_total)
    final_train_acc = running_correct / max(1, running_total)
    final_val_acc = val_accs[-1] if val_accs else 0.0

    return TrainResult(
        seed=cfg.seed,
        steps=cfg.steps,
        wall_time_s=wall,
        steps_per_sec=cfg.steps / wall,
        final_train_loss=final_train_loss,
        final_train_acc=final_train_acc,
        best_val_acc=best_val_acc,
        final_val_acc=final_val_acc,
        val_accs=val_accs,
        config=asdict(cfg),
    )


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="ATLAS baseline trainer")
    p.add_argument("--grad_centralize", action="store_true", default=False)
    for f_name, f_type in [
        ("seed", int),
        ("steps", int),
        ("batch_size", int),
        ("lr", float),
        ("weight_decay", float),
        ("hidden_dim", int),
        ("dropout", float),
        ("label_smoothing", float),
        ("warmup_steps", int),
        ("grad_clip", float),
        ("init_scale", float),
        ("ema_decay", float),
        ("mixup_alpha", float),
        ("cutout_size", int),
        ("num_workers", int),
        ("log_every", int),
        ("eval_every", int),
    ]:
        p.add_argument(f"--{f_name}", type=f_type, default=getattr(TrainConfig(), f_name))
    for f_name in ["optimizer", "scheduler", "activation", "norm", "mixing", "data_dir"]:
        p.add_argument(f"--{f_name}", type=str, default=getattr(TrainConfig(), f_name))
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    result = train(cfg)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
