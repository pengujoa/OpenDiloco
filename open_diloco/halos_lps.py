"""
HALoS Local Parameter Server (LPS).

Manages a local model copy, applies momentum-based updates from worker
pseudo-gradients, and communicates asynchronously with the GPS.

Algorithm:
    1. Worker trains H local steps → pseudo_grad = (start_params − final_params)
    2. LPS receives pseudo_grad → DelayedNesterov update (β_l=0.9, d_l=1)
    3. Every K updates → send accumulated delta to GPS (non-blocking)
    4. When GPS responds → merge: model = (1−α)·local + α·global
"""

import logging
import socket
import threading
from typing import Optional

import torch

try:
    from halos_delayed_nesterov import DelayedNesterovOptimizer
    from halos_gps import send_tensors, recv_tensors
except ImportError:
    from open_diloco.halos_delayed_nesterov import DelayedNesterovOptimizer
    from open_diloco.halos_gps import send_tensors, recv_tensors

log = logging.getLogger("halos.lps")


class LocalParameterServer:
    """
    Lifecycle per round:
        lps_params = lps.get_model_params()  → worker copies these to start training
        pseudo_grad = old_params - new_params → after H local steps
        lps.step(pseudo_grad)                 → momentum update + maybe GPS sync
    """

    def __init__(
        self,
        model_params: list[torch.Tensor],
        gps_host: str,
        gps_port: int,
        K: int = 2,
        alpha: float = 0.25,
        lr: float = 0.2,
        beta: float = 0.9,
        buffer_size: int = 1,
        c: float = 0.0,
    ):
        self.params = [torch.nn.Parameter(p.clone().cpu()) for p in model_params]
        self.optimizer = DelayedNesterovOptimizer(
            self.params, lr=lr, beta=beta, c=c, buffer_size=buffer_size
        )
        self.K = K
        self.alpha = alpha
        self._n = 0

        # Snapshot at last GPS merge — used to compute accumulated delta
        self._merged_snap = [p.data.clone() for p in self.params]

        # GPS TCP connection (lazy, persistent, background I/O)
        self._gps_addr = (gps_host, gps_port)
        self._sock: Optional[socket.socket] = None
        self._io_lock = threading.Lock()

        # Pending global model from GPS (applied before next worker interaction)
        self._pending: Optional[list[torch.Tensor]] = None
        self._pend_lock = threading.Lock()

        log.info(
            f"LPS init: K={K}, α={alpha}, lr={lr}, β={beta}, d_l={buffer_size}, "
            f"GPS={gps_host}:{gps_port}"
        )

    # ── Public API ───────────────────────────────────────────────────────

    def step(self, pseudo_grad: list[torch.Tensor]):
        """
        Process one worker pseudo-gradient.

        Applies momentum update to local model. Every K calls, sends
        accumulated delta to GPS in a background thread (non-blocking).
        """
        self._try_merge()

        self.optimizer.zero_grad()
        for p, g in zip(self.params, pseudo_grad):
            p.grad = g.to(p.device)
        self.optimizer.step()
        self._n += 1
        log.info(f"LPS update #{self._n} (K={self.K})")

        if self._n % self.K == 0:
            delta = [s - p.data for s, p in zip(self._merged_snap, self.params)]
            delta_norm = sum(d.norm().item() ** 2 for d in delta) ** 0.5
            log.info(f"Sending accumulated delta to GPS (n={self._n}, delta_norm={delta_norm:.4f})")
            threading.Thread(
                target=self._gps_exchange, args=(delta,), daemon=True
            ).start()

    def get_model_params(self) -> list[torch.Tensor]:
        """Current LPS model for worker to start a new H-step round."""
        self._try_merge()
        return [p.data.clone() for p in self.params]

    # ── GPS communication (background thread) ────────────────────────────

    def _gps_exchange(self, delta: list[torch.Tensor]):
        """Send accumulated delta to GPS, receive updated global model."""
        try:
            with self._io_lock:
                self._connect()
                send_tensors(self._sock, delta)
                gp = recv_tensors(self._sock)
            if gp is not None:
                with self._pend_lock:
                    self._pending = gp
                log.info("Received global model from GPS")
        except Exception as e:
            log.warning(f"GPS comm error: {e}")
            self._sock = None

    def _connect(self):
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect(self._gps_addr)
            log.info(f"Connected to GPS at {self._gps_addr}")

    def _try_merge(self):
        """Apply pending GPS merge: model = (1−α)·local + α·global."""
        with self._pend_lock:
            gp = self._pending
            self._pending = None
        if gp is None:
            return
        for p, g in zip(self.params, gp):
            p.data.mul_(1.0 - self.alpha).add_(g.to(p.device), alpha=self.alpha)
        self._merged_snap = [p.data.clone() for p in self.params]
        log.info(f"Merged GPS model (α={self.alpha}, n={self._n})")

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
