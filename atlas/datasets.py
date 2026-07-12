"""Char-level datasets for tinyshakespeare LM experiments."""

from __future__ import annotations

from pathlib import Path

import torch


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class CharDataset:
    """Character dataset with optional shared vocabulary."""

    def __init__(
        self,
        text: str,
        *,
        stoi: dict[str, int] | None = None,
        itos: dict[int, str] | None = None,
    ) -> None:
        if stoi is None:
            stoi, itos = build_vocab(text)
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)
        self.data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        stoi: dict[str, int],
        itos: dict[int, str],
    ) -> CharDataset:
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, stoi=stoi, itos=itos)


def load_train_val(
    data_path: Path | str,
    train_frac: float = 0.9,
) -> tuple[CharDataset, CharDataset]:
    """Load train/val splits with vocabulary built from the full corpus."""
    text = Path(data_path).read_text(encoding="utf-8")
    split = int(train_frac * len(text))
    stoi, itos = build_vocab(text)
    train_ds = CharDataset(text[:split], stoi=stoi, itos=itos)
    val_ds = CharDataset(text[split:], stoi=stoi, itos=itos)
    assert train_ds.stoi == val_ds.stoi
    assert train_ds.vocab_size == val_ds.vocab_size
    return train_ds, val_ds


def random_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous chunks (high token diversity)."""
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def consecutive_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Buggy consecutive batches (for regression tests only)."""
    x = torch.stack([data[start + i : start + i + block_size] for i in range(batch_size)])
    y = torch.stack([data[start + i + 1 : start + i + block_size + 1] for i in range(batch_size)])
    return x, y
