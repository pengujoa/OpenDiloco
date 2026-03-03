"""
Lion optimizer implementation from:
"Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

Reference: https://arxiv.org/abs/2302.06675
"""

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer.

    Uses sign-based updates with interpolated momentum, resulting in
    uniform update magnitudes and reduced memory usage compared to Adam
    (only one momentum buffer instead of two).

    Recommended hyperparameter adjustments vs AdamW:
      - lr:           3~10x smaller (e.g. 1e-4 instead of 3e-4)
      - weight_decay: 3~10x larger  (e.g. 0.3~1.0 instead of 0.1)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

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

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # Update: sign of interpolation between momentum and gradient
                update = exp_avg.lerp(grad, 1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # Momentum update (uses beta2, different from the update step's beta1)
                exp_avg.lerp_(grad, 1.0 - beta2)

        return loss
