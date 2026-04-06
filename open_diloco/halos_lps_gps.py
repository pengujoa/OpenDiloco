"""
HALoS-style LPS + GPS state (utnslab/halos semantics, collapsed onto DiLoCo messengers).

- LPS: Delayed Nesterov on regional-mean pseudo-grad (worker contribution) each outer step.
- Every K LPS steps: tensors fed to hivemind = (lps_ref - lps_theta) (GPS input).
- GPS: hivemind-mean delta → Delayed Nesterov on gps_theta, then
      lps_theta <- (1-alpha)*lps_theta + alpha*gps_theta, lps_ref <- lps_theta.

Outer DiLoCo optimizer step is bypassed when this runtime is active; offloaded weights
are overwritten from lps_theta after each outer epoch.

Does not implement separate worker processes or bandwidth modeling.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from halos_delayed_nesterov import DelayedNesterovOptimizer
except ImportError:
    from open_diloco.halos_delayed_nesterov import DelayedNesterovOptimizer  # type: ignore

try:
    from halos_gps_tcp import HalosRemoteGpsClient
except ImportError:
    from open_diloco.halos_gps_tcp import HalosRemoteGpsClient  # type: ignore

logger = logging.getLogger(__name__)


class HalosLpsGpsRuntime:
    def __init__(
        self,
        *,
        reference_params: Sequence[torch.nn.Parameter],
        K: int,
        alpha: float,
        lps_lr: float,
        lps_beta: float,
        lps_buffer_size: int,
        lps_c: float,
        gps_lr: float,
        gps_beta: float,
        gps_buffer_size: int,
        gps_c: float,
        remote_gps_host: Optional[str] = None,
        remote_gps_port: Optional[int] = None,
        remote_gps_timeout: float = 600.0,
        remote_gps_init_rank: int = 0,
        remote_gps_async: bool = False,
    ):
        if K < 1:
            raise ValueError("K (halos_local_updates_accumulation) must be >= 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha (halos_model_merge_weight) must be in [0, 1]")

        self.K = K
        self.alpha = alpha
        self.lps_step_count = 0

        self.lps_params = nn.ParameterList(nn.Parameter(p.detach().clone()) for p in reference_params)
        self.gps_params = nn.ParameterList(nn.Parameter(p.detach().clone()) for p in reference_params)
        self.lps_ref: List[torch.Tensor] = [p.detach().clone() for p in self.lps_params]

        self.lps_opt = DelayedNesterovOptimizer(
            list(self.lps_params),
            lr=lps_lr,
            beta=lps_beta,
            buffer_size=lps_buffer_size,
            c=lps_c,
        )
        self.gps_opt = DelayedNesterovOptimizer(
            list(self.gps_params),
            lr=gps_lr,
            beta=gps_beta,
            buffer_size=gps_buffer_size,
            c=gps_c,
        )

        self.use_remote_gps = bool(remote_gps_host and remote_gps_port is not None)
        self.remote_gps_init_rank = int(remote_gps_init_rank)
        self.remote_gps_async = bool(remote_gps_async)
        self._remote_client: Optional[HalosRemoteGpsClient] = None
        self._sent_remote_init = False
        self._remote_exchange_executor = None
        if self.use_remote_gps:
            self._remote_client = HalosRemoteGpsClient(
                remote_gps_host,
                int(remote_gps_port),
                timeout=float(remote_gps_timeout),
            )
            if self.remote_gps_async:
                from concurrent.futures import ThreadPoolExecutor

                self._remote_exchange_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="halos_remote_gps"
                )
            logger.info(
                "[HALoS LPS/GPS] Remote GPS enabled → %s:%s (init_rank=%s async_io=%s)",
                remote_gps_host,
                remote_gps_port,
                self.remote_gps_init_rank,
                self.remote_gps_async,
            )

        logger.info(
            "[HALoS LPS/GPS] Initialized: K=%s alpha=%s | LPS lr=%s beta=%s buf=%s | GPS lr=%s beta=%s buf=%s",
            K,
            alpha,
            lps_lr,
            lps_beta,
            lps_buffer_size,
            gps_lr,
            gps_beta,
            gps_buffer_size,
        )

    def will_run_gps_after_this_lps(self) -> bool:
        return (self.lps_step_count + 1) % self.K == 0

    @torch.no_grad()
    def apply_lps_round(self, tensors: List[torch.Tensor]) -> None:
        """tensors: regional-mean pseudo-grad (same layout as grad averager)."""
        will_gps = self.will_run_gps_after_this_lps()

        self.lps_opt.zero_grad(set_to_none=True)
        for i, t in enumerate(tensors):
            p = self.lps_params[i]
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            p.grad.copy_(t.to(device=p.data.device, dtype=p.data.dtype, non_blocking=True))
        self.lps_opt.step()
        self.lps_opt.zero_grad(set_to_none=True)
        self.lps_step_count += 1

        if will_gps:
            for i, t in enumerate(tensors):
                delta = self.lps_ref[i].to(t.device, dtype=t.dtype) - self.lps_params[i].data.to(
                    device=t.device, dtype=t.dtype
                )
                t.copy_(delta)
            logger.info(
                "[HALoS LPS/GPS] LPS step %s → GPS round (delta to %s)",
                self.lps_step_count,
                "remote GPS" if self.use_remote_gps else "hivemind",
            )
        else:
            logger.debug(
                "[HALoS LPS/GPS] LPS-only outer step (%s/%s)",
                self.lps_step_count % self.K,
                self.K,
            )

    @torch.no_grad()
    def after_gps_allreduce(self, averaged_delta_tensors: Sequence[torch.Tensor]) -> None:
        """averaged_delta_tensors: hivemind output (same order as tensors)."""
        self.gps_opt.zero_grad(set_to_none=True)
        for i, t in enumerate(averaged_delta_tensors):
            p = self.gps_params[i]
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            p.grad.copy_(t.to(device=p.data.device, dtype=p.data.dtype, non_blocking=True))
        self.gps_opt.step()
        self.gps_opt.zero_grad(set_to_none=True)

        for i in range(len(self.lps_params)):
            lp = self.lps_params[i].data
            gp = self.gps_params[i].data.to(lp.device, dtype=lp.dtype)
            lp.mul_(1.0 - self.alpha).add_(gp, alpha=self.alpha)
            self.lps_ref[i] = lp.detach().clone()

        logger.info("[HALoS LPS/GPS] GPS step + merge (alpha=%s); lps_ref updated", self.alpha)

    def _exchange_remote_with_retry(
        self, deltas_cpu: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], int]:
        assert self._remote_client is not None
        last_err: Optional[BaseException] = None
        for _ in range(3000):
            try:
                if self._remote_exchange_executor is not None:
                    fut = self._remote_exchange_executor.submit(self._remote_client.exchange, deltas_cpu)
                    return fut.result(timeout=self._remote_client.timeout)
                return self._remote_client.exchange(deltas_cpu)
            except RuntimeError as e:
                last_err = e
                msg = str(e).lower()
                if "not initialized" in msg or "gps not initialized" in msg:
                    time.sleep(0.2)
                    continue
                raise
        raise RuntimeError(f"Remote GPS exchange failed after retries: {last_err}")

    @torch.no_grad()
    def after_remote_global_from_server(self, global_tensors: Sequence[torch.Tensor]) -> None:
        """GPS step already applied on the server; mirror globals and merge into LPS."""
        for i, g in enumerate(global_tensors):
            src = g.to(device=self.gps_params[i].data.device, dtype=self.gps_params[i].data.dtype)
            self.gps_params[i].data.copy_(src)

        for i in range(len(self.lps_params)):
            lp = self.lps_params[i].data
            gp = self.gps_params[i].data.to(lp.device, dtype=lp.dtype)
            lp.mul_(1.0 - self.alpha).add_(gp, alpha=self.alpha)
            self.lps_ref[i] = lp.detach().clone()

        logger.info(
            "[HALoS LPS/GPS] Remote GPS merge (alpha=%s); lps_ref updated (no hivemind GPS allreduce)",
            self.alpha,
        )

    @torch.no_grad()
    def run_remote_gps_round(self, delta_tensors: Sequence[torch.Tensor], world_rank: int) -> None:
        """Send local accumulated delta to the TCP GPS process and apply returned global weights."""
        if not self.use_remote_gps or self._remote_client is None:
            raise RuntimeError("run_remote_gps_round called without remote GPS configured")
        deltas_cpu = [t.detach().cpu() for t in delta_tensors]
        if world_rank == self.remote_gps_init_rank and not self._sent_remote_init:
            tpl = [p.detach().cpu().clone() for p in self.lps_params]
            self._remote_client.init_server(tpl)
            self._sent_remote_init = True
        globs, ver = self._exchange_remote_with_retry(deltas_cpu)
        _ = ver
        self.after_remote_global_from_server(globs)

    @torch.no_grad()
    def sync_lps_to_outer(self, outer_params: Sequence[torch.nn.Parameter]) -> None:
        for op, lp in zip(outer_params, self.lps_params):
            op.data.copy_(lp.data.to(op.device, dtype=op.dtype, non_blocking=True))
