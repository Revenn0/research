"""Minimal char-level transformer with global attention or causal long-conv FFT mixing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

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


class LongConvFFT(nn.Module):
    """Causal depthwise long convolution via FFT (pad to 2T-1, truncate to T)."""

    def __init__(self, n_embd: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = nn.Parameter(torch.randn(n_embd, kernel_size) * 0.02)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x_ch = x.transpose(1, 2)
        n = 2 * t - 1
        x_pad = F.pad(x_ch, (0, t - 1))
        k = self.filters[:, : min(self.kernel_size, t)]
        k_pad = F.pad(k, (0, n - k.shape[-1]))
        x_f = torch.fft.rfft(x_pad, n=n)
        k_f = torch.fft.rfft(k_pad, n=n).unsqueeze(0)
        y_f = x_f * k_f
        y = torch.fft.irfft(y_f, n=n)[..., :t]
        y = y.transpose(1, 2)
        return self.dropout(self.proj(y))


def build_norm(name: str, n_embd: int) -> nn.Module | None:
    if name == "none":
        return None
    if name == "layernorm":
        return nn.LayerNorm(n_embd)
    raise ValueError(f"Unknown norm: {name}")


def build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class Block(nn.Module):
    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        mixing: str,
        norm: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.ln1 = build_norm(norm, n_embd)
        self.ln2 = build_norm(norm, n_embd)
        if mixing == "global-attn":
            self.mixer = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        elif mixing == "long-conv-fft":
            self.mixer = LongConvFFT(n_embd, kernel_size=block_size, dropout=dropout)
        else:
            raise ValueError(f"Unknown mixing: {mixing}")
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            build_activation(activation),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def _norm(self, ln: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
        return ln(x) if ln is not None else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self._norm(self.ln1, x))
        x = x + self.ff(self._norm(self.ln2, x))
        return x


class CharLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        mixing: str = "global-attn",
        norm: str = "layernorm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                    mixing=mixing,
                    norm=norm,
                    activation=activation,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = build_norm(norm, n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        if self.ln_f is not None:
            x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
