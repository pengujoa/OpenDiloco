"""
Delayed Nesterov outer optimizer used in HALoS (ICML 2025) reference code.

Implementation matches `utnslab/halos` `src/halos/opt.py` (DelayedNesterovOptimizer),
which cites: https://arxiv.org/pdf/2401.09135

Intended use: replace DiLoCo's outer `torch.optim.SGD(..., nesterov=True)` when
reproducing HALoS GPS (global parameter server) updates. Hyperparameters from
`examples/run_halos.sh` (after shell scaling):

  _GLR = GLR * GMUD   # e.g. 0.15 * 2 = 0.3
  GBETA = 0.5, GMUD = buffer_size = 2
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class DelayedNesterovOptimizer(Optimizer):
    """
    Bufferized delayed Nesterov update (same update rule as official HALoS repo).

    Expects `.grad` on each parameter to be set before `step()` (as DiLoCo does
    when loading averaged pseudo-gradients into the outer optimizer).
    """

    def __init__(
        self,
        params,
        lr: float,
        beta: float,
        c: float = 0.0,
        buffer_size: int = 1,
    ):
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.beta = beta
        self.c = c
        self.buffer_size = buffer_size
        self._t = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p] = {
                        "m": torch.zeros_like(p, memory_format=torch.preserve_format),
                        "delta": torch.zeros_like(p, memory_format=torch.preserve_format),
                    }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state["m"]
                delta = state["delta"]
                grad = p.grad

                delta.add_(grad)

                if (self._t + 1) % self.buffer_size == 0:
                    m.mul_(self.beta).add_(delta, alpha=1.0 / self.buffer_size)
                    coeff_m = (1.0 - self.c * self.buffer_size + self.c) * self.beta
                    p.add_(m, alpha=-lr * coeff_m)
                    p.add_(grad, alpha=-lr / self.buffer_size)
                    delta.zero_()
                else:
                    p.add_(m, alpha=-lr * self.c * self.beta)
                    p.add_(grad, alpha=-lr / self.buffer_size)

        self._t += 1
        return loss

    def state_dict(self):
        d = super().state_dict()
        d["delayed_nesterov_step"] = self._t
        return d

    def load_state_dict(self, state_dict):
        self._t = int(state_dict.pop("delayed_nesterov_step", 0))
        super().load_state_dict(state_dict)
