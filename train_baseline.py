#!/usr/bin/env python3
"""Trainer de referência mínimo ATLAS — patches aplicados via flags, nunca reescrita total."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    norm: str = "none"  # none | batchnorm | layernorm | rms_free
    mixing: str = "none"  # none | local_blend | local_blend_k5 | multi_scale_blend | tri_scale_blend | cosine_gate_blend | entropy_gate_blend | post_act_blend | cascade_blend
    arch: str = "cnn"  # cnn | vit | hybrid_blend
    vit_dim: int = 64
    vit_depth: int = 2
    vit_heads: int = 4
    vit_mixer: str = "attn"  # attn | token_blend | token_blend_ms
    init_scale: float = 1.0
    ema_decay: float = 0.0
    mixup_alpha: float = 0.0
    cutout_size: int = 0
    # escala H100
    device: str = "auto"  # auto | cpu | cuda
    amp: bool = False  # bf16 autocast (GPU)
    width_mult: float = 1.0  # multiplica canais do SmallCNN (baselines maiores)
    compile: bool = False  # torch.compile
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
    """NOVEL: mistura local depthwise com gate escalar por canal."""

    def __init__(self, channels: int, kernel: int = 3, gate_mode: str = "mean") -> None:
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(channels, channels, kernel, padding=pad, groups=channels, bias=False)
        if gate_mode == "laplacian":
            w = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            self.dw.weight.data = w.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        else:
            nn.init.dirac_(self.dw.weight)
        self.gate_mode = gate_mode

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "mean":
            return torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        if self.gate_mode == "var":
            return torch.sigmoid(x.var(dim=(2, 3), keepdim=True, unbiased=False) * 4.0)
        if self.gate_mode == "dual":
            m = x.mean(dim=(2, 3), keepdim=True)
            v = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            return torch.sigmoid(m) * torch.sigmoid(v * 4.0)
        return torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw(x)
        gate = self._gate(x)
        return gate * local + (1.0 - gate) * x


class MultiScaleBlend(nn.Module):
    """NOVEL: fusão paralela depthwise 3x3 + 5x5 com gate de média espacial."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        nn.init.dirac_(self.dw3.weight)
        nn.init.dirac_(self.dw5.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = 0.5 * self.dw3(x) + 0.5 * self.dw5(x)
        gate = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * x


class TriScaleBlend(nn.Module):
    """NOVEL: fusão paralela depthwise 3x3 + 5x5 + 7x7 com gate de média espacial."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        for m in (self.dw3, self.dw5, self.dw7):
            nn.init.dirac_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = (self.dw3(x) + self.dw5(x) + self.dw7(x)) / 3.0
        gate = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * x


class CosineGateBlend(nn.Module):
    """NOVEL: gate = sigmoid(cossim espacial entre x e conv(x)) por canal."""

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


class EntropyGateBlend(nn.Module):
    """NOVEL: gate derivado de entropia espacial da magnitude por canal."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw(x)
        mag = x.abs()
        p = mag / mag.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        ent = -(p * (p + 1e-8).log()).sum(dim=(2, 3), keepdim=True)
        gate = torch.sigmoid(ent * 2.0)
        return gate * local + (1.0 - gate) * x


class PostActBlend(nn.Module):
    """NOVEL: blend APÓS ReLU — mixing no espaço de ativação."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.relu(x)
        local = self.dw(activated)
        gate = torch.sigmoid(activated.mean(dim=(2, 3), keepdim=True))
        return gate * local + (1.0 - gate) * activated


class CascadeBlend(nn.Module):
    """NOVEL: dois blends em série — refinamento progressivo."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blend1 = LocalBlend(channels, kernel=3, gate_mode="mean")
        self.blend2 = LocalBlend(channels, kernel=3, gate_mode="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blend2(self.blend1(x))


def make_mixing(name: str, channels: int) -> nn.Module:
    if name == "local_blend":
        return LocalBlend(channels, kernel=3, gate_mode="mean")
    if name == "local_blend_var":
        return LocalBlend(channels, kernel=3, gate_mode="var")
    if name == "local_blend_k5":
        return LocalBlend(channels, kernel=5, gate_mode="mean")
    if name == "laplacian_blend":
        return LocalBlend(channels, kernel=3, gate_mode="laplacian")
    if name == "dual_gate_blend":
        return LocalBlend(channels, kernel=3, gate_mode="dual")
    if name == "multi_scale_blend":
        return MultiScaleBlend(channels)
    if name == "tri_scale_blend":
        return TriScaleBlend(channels)
    if name == "cosine_gate_blend":
        return CosineGateBlend(channels)
    if name == "entropy_gate_blend":
        return EntropyGateBlend(channels)
    if name == "post_act_blend":
        return PostActBlend(channels)
    if name == "cascade_blend":
        return CascadeBlend(channels)
    return nn.Identity()


class RMSFree(nn.Module):
    """NOVEL: RMS sem parâmetros afim — escala por canal."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            return x
        rms = x.pow(2).mean(dim=(2, 3), keepdim=True).sqrt().clamp(min=1e-5)
        return x / rms


class TokenBlend(nn.Module):
    """NOVEL: token mixing sem atenção — blend depthwise 1D sobre tokens com gate global.

    Aplica a descoberta local_blend ao espaço de tokens de um Transformer:
    mistura local de tokens vizinhos (conv1d depthwise) + gate escalar por dim
    derivado da média sobre tokens. Zero matrizes QKV — O(N) em vez de O(N²).
    """

    def __init__(self, dim: int, kernel: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim, bias=False)
        nn.init.dirac_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        h = x.transpose(1, 2)  # (B, D, N)
        local = self.dw(h).transpose(1, 2)
        gate = torch.sigmoid(x.mean(dim=1, keepdim=True))  # (B, 1, D)
        return gate * local + (1.0 - gate) * x


class TokenBlendMS(nn.Module):
    """NOVEL: multi-scale token blend — kernels 3 e 5 sobre a sequência de tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim, bias=False)
        nn.init.dirac_(self.dw3.weight)
        nn.init.dirac_(self.dw5.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        local = (0.5 * self.dw3(h) + 0.5 * self.dw5(h)).transpose(1, 2)
        gate = torch.sigmoid(x.mean(dim=1, keepdim=True))
        return gate * local + (1.0 - gate) * x


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mixer: str) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer_name = mixer
        if mixer == "attn":
            self.mixer = nn.MultiheadAttention(dim, heads, batch_first=True)
        elif mixer == "token_blend":
            self.mixer = TokenBlend(dim)
        else:
            self.mixer = TokenBlendMS(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.mixer_name == "attn":
            h, _ = self.mixer(h, h, h, need_weights=False)
        else:
            h = self.mixer(h)
        x = x + h
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """ViT mínimo para FashionMNIST: patch 4x4, budget compatível com SmallCNN."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        dim = cfg.vit_dim
        self.patch = nn.Conv2d(1, dim, 4, stride=4)  # 28->7, 49 tokens
        n_tokens = 49
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, cfg.vit_heads, cfg.vit_mixer) for _ in range(cfg.vit_depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)  # (B, 49, D)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))


class HybridBlendNet(nn.Module):
    """NOVEL revolucionário: CNN multi-scale spatial blend + TokenBlend sobre tokens espaciais.

    Fusão de duas descobertas ATLAS:
    1) tri_scale/multi_scale depthwise blending (espacial, por canal)
    2) TokenBlend O(N) sobre mapa de features como sequência de tokens (7x7=49)

    Diferença vs ViT: features CNN ricas antes do token mixing.
    Diferença vs CNN puro: mixing global entre posições espaciais sem self-attention O(N²).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        c1, c2 = 32, 64
        mix = cfg.mixing if cfg.mixing != "none" else "tri_scale_blend"
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.mix1 = make_mixing(mix, c1)
        self.mix2 = make_mixing(mix, c2)
        self.token_mix = TokenBlendMS(c2)
        self.norm_tok = nn.LayerNorm(c2)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(c2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.mix1(self.conv1(x))))
        x = self.pool(F.relu(self.mix2(self.conv2(x))))  # (B, 64, 7, 7)
        tokens = x.flatten(2).transpose(1, 2)  # (B, 49, 64)
        tokens = tokens + self.token_mix(self.norm_tok(tokens))
        x = self.dropout(tokens.mean(dim=1))
        return self.fc(x)


class SmallCNN(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.activation_name = cfg.activation
        self.mixing_name = cfg.mixing
        use_builtin_act = cfg.activation in ("relu", "gelu", "silu")
        c1, c2 = int(32 * cfg.width_mult), int(64 * cfg.width_mult)
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = self._make_norm(cfg.norm, c1)
        self.norm2 = self._make_norm(cfg.norm, c2)
        self.mix1 = make_mixing(cfg.mixing, c1)
        self.mix2 = make_mixing(cfg.mixing, c2)
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
        if norm == "rms_free":
            return RMSFree()
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
        h1 = self.norm1(self.conv1(x))
        if self.mixing_name == "post_act_blend":
            x = self.pool(self.mix1(h1))
        else:
            x = self.pool(self._activate(self.mix1(h1)))
        h2 = self.norm2(self.conv2(x))
        if self.mixing_name == "post_act_blend":
            x = self.pool(self.mix2(h2))
        else:
            x = self.pool(self._activate(self.mix2(h2)))
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
    pin = resolve_device(cfg.device).type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "norm_feedback":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "grad_shrink":
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


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.arch == "vit":
        return TinyViT(cfg)
    if cfg.arch == "hybrid_blend":
        return HybridBlendNet(cfg)
    return SmallCNN(cfg)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(cfg: TrainConfig, progress_path: str = "") -> TrainResult:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile:
        model = torch.compile(model)
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            if cfg.mixup_alpha > 0:
                loss = lam * F.cross_entropy(logits, y_a, label_smoothing=cfg.label_smoothing) + (
                    1 - lam
                ) * F.cross_entropy(logits, y_b, label_smoothing=cfg.label_smoothing)
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

        loss.backward()
        if cfg.optimizer == "grad_shrink":
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(torch.sigmoid(p.grad.data.abs()))
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

        if progress_path and (step % 25 == 0 or step == cfg.steps):
            _write_progress(progress_path, cfg, step, best_val_acc, val_accs)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            eval_model = model
            if ema is not None:
                eval_model = build_model(cfg).to(device)
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


def _write_progress(
    path: str, cfg: TrainConfig, step: int, best_val_acc: float, val_accs: list[float]
) -> None:
    payload = {
        "seed": cfg.seed,
        "step": step,
        "total_steps": cfg.steps,
        "step_pct": round(100.0 * step / max(1, cfg.steps), 1),
        "best_val_acc": best_val_acc,
        "last_val_acc": val_accs[-1] if val_accs else None,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> tuple[TrainConfig, str, str]:
    p = argparse.ArgumentParser(description="ATLAS baseline trainer")
    p.add_argument("--grad_centralize", action="store_true", default=False)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
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
        ("width_mult", float),
        ("num_workers", int),
        ("log_every", int),
        ("eval_every", int),
    ]:
        p.add_argument(f"--{f_name}", type=f_type, default=getattr(TrainConfig(), f_name))
    p.add_argument("--result_path", type=str, default="", help="opcional: grava JSON do resultado ao terminar")
    p.add_argument("--progress_path", type=str, default="", help="opcional: grava progresso ao vivo (steps)")
    for f_name in ["optimizer", "scheduler", "activation", "norm", "mixing", "data_dir", "arch", "vit_mixer", "device"]:
        p.add_argument(f"--{f_name}", type=str, default=getattr(TrainConfig(), f_name))
    for f_name in ["vit_dim", "vit_depth", "vit_heads"]:
        p.add_argument(f"--{f_name}", type=int, default=getattr(TrainConfig(), f_name))
    args = p.parse_args()
    d = vars(args)
    result_path = d.pop("result_path", "")
    progress_path = d.pop("progress_path", "")
    return TrainConfig(**d), result_path, progress_path


def main() -> None:
    cfg, result_path, progress_path = parse_args()
    result = train(cfg, progress_path=progress_path)
    payload = asdict(result)
    print(json.dumps(payload, indent=2))
    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
