"""Perplexity evaluation on a fixed WikiText-2 slice."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


LOCAL_FALLBACK_TEXT = """\
WikiText fallback corpus for KappaQ mini-test when HuggingFace download fails.
The quick brown fox jumps over the lazy dog. Machine learning models learn patterns from data.
Quantization reduces memory by storing weights with fewer bits while preserving accuracy when possible.
Orthogonal rotations can reshape weight distributions before rounding to a discrete grid.
Mean squared error scale search finds clip factors that minimize reconstruction error on weights.
Perplexity measures how well a language model predicts the next token in a sequence.
Natural language processing spans translation summarization question answering and generation.
Transformers use attention to mix information across positions in the context window.
Dense projections include query key value output and MLP up gate down matrices.
Embeddings layer norms and the language model head often remain in higher precision.
Group size one hundred twenty eight is a common packing choice for low bit quantization.
Symmetric three bit grids map values onto seven reconstruction levels including zero.
Seeds control stochastic rotations so experiments can be repeated exactly.
CPU only environments require smaller models and shorter evaluation slices.
This paragraph is repeated padding so the local corpus yields enough tokens for chunking.
""" * 40


def corpus_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_wikitext_text() -> Tuple[str, str, str]:
    """Returns (text, source, hash16)."""
    errors = []
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset

            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([t for t in ds["text"] if t and t.strip()])
            if len(text) < 1000:
                raise RuntimeError("wikitext too short")
            return text, f"hf:{repo}/wikitext-2-raw-v1/test", corpus_hash(text)
        except Exception as e:
            errors.append(f"{repo}:{type(e).__name__}:{e}")
    text = LOCAL_FALLBACK_TEXT
    return text, f"local_fallback:{' | '.join(errors)}", corpus_hash(text)


def build_eval_batches(
    tokenizer,
    text: str,
    n_sequences: int = 64,
    max_length: int = 256,
) -> Tuple[List[torch.Tensor], int]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    # drop trailing incomplete chunk for fair fixed token count
    usable = (ids.numel() // max_length) * max_length
    ids = ids[:usable]
    chunks = list(ids.split(max_length))
    if len(chunks) < n_sequences:
        # wrap if corpus short
        while len(chunks) < n_sequences and chunks:
            chunks.extend(chunks[: max(1, n_sequences - len(chunks))])
    chunks = chunks[:n_sequences]
    n_tokens = sum(int(c.numel()) for c in chunks)
    return chunks, n_tokens


@torch.no_grad()
def eval_perplexity(
    model: nn.Module,
    batches: List[torch.Tensor],
    device: torch.device,
) -> Tuple[float, int, float]:
    """Returns (ppl, n_tokens, wall_s)."""
    import time

    model.eval()
    loss_fn = CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tok = 0
    t0 = time.perf_counter()
    for chunk in batches:
        input_ids = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_nll += float(nll.item())
        total_tok += int(shift_labels.numel())
    wall = time.perf_counter() - t0
    mean_nll = total_nll / max(total_tok, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    return ppl, total_tok, wall
