"""Optimizers for ATLAS LM experiments."""

from __future__ import annotations

import torch


class SignLion(torch.optim.Optimizer):
    """Lion-style signed-momentum optimizer (SignLion)."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                c = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
                p.add_(torch.sign(c), alpha=-lr)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                exp_avg.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "signlion":
        return SignLion(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
