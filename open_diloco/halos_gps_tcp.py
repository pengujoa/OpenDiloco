"""
TCP protocol for a dedicated HALoS GPS process.

LPS (training / hivemind messenger) sends accumulated deltas; the GPS process
applies Delayed Nesterov on its global copy and returns updated global weights.
No decentralized allreduce among LPS nodes is required for the GPS step.
"""

from __future__ import annotations

import io
import logging
import socket
import struct
import threading
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MAGIC = b"HGPS"
# wire format: MAGIC (4) + uint32 msg_type + uint64 payload_len + payload

MSG_INIT = 1
MSG_EXCHANGE = 2
MSG_ACK = 3
MSG_RESPONSE = 4
MSG_ERROR = 5
MSG_PING = 6
MSG_PONG = 7


def _send_all(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    while len(view):
        n = sock.send(view)
        view = view[n:]


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: List[bytes] = []
    remaining = n
    while remaining > 0:
        b = sock.recv(min(65536, remaining))
        if not b:
            raise ConnectionError("socket closed while receiving")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


def send_message(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    header = struct.pack("<IQ", msg_type, len(payload))
    _send_all(sock, MAGIC + header)
    _send_all(sock, payload)


def recv_message(sock: socket.socket) -> Tuple[int, bytes]:
    head = _recv_exact(sock, 16)
    if head[:4] != MAGIC:
        raise ValueError(f"bad magic: {head[:4]!r}")
    msg_type, payload_len = struct.unpack("<IQ", head[4:16])
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    return int(msg_type), payload


def _torch_save_bytes(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _torch_load_bytes(data: bytes) -> Any:
    buf = io.BytesIO(data)
    try:
        return torch.load(buf, map_location="cpu", weights_only=False)
    except TypeError:
        buf.seek(0)
        return torch.load(buf, map_location="cpu")


class HalosGpsTcpState:
    """In-process GPS state (used by the standalone TCP server)."""

    def __init__(
        self,
        *,
        gps_lr: float,
        gps_beta: float,
        gps_buffer_size: int,
        gps_c: float,
    ):
        self._lock = threading.Lock()
        self.gps_lr = gps_lr
        self.gps_beta = gps_beta
        self.gps_buffer_size = gps_buffer_size
        self.gps_c = gps_c
        self.gps_params: Optional[nn.ParameterList] = None
        self.gps_opt = None
        self.version = 0

    def init_from_tensors(self, tensors: Sequence[torch.Tensor]) -> None:
        try:
            from halos_delayed_nesterov import DelayedNesterovOptimizer
        except ImportError:
            from open_diloco.halos_delayed_nesterov import DelayedNesterovOptimizer  # type: ignore

        with self._lock:
            self.gps_params = nn.ParameterList(nn.Parameter(t.detach().float().clone()) for t in tensors)
            self.gps_opt = DelayedNesterovOptimizer(
                list(self.gps_params),
                lr=self.gps_lr,
                beta=self.gps_beta,
                buffer_size=self.gps_buffer_size,
                c=self.gps_c,
            )
            self.version = 0
            logger.info("[GPS TCP] Initialized global model from client (%d tensors).", len(self.gps_params))

    @torch.no_grad()
    def apply_delta_and_return_global(self, deltas: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], int]:
        with self._lock:
            if self.gps_params is None or self.gps_opt is None:
                raise RuntimeError("GPS not initialized; send MSG_INIT first.")
            if len(deltas) != len(self.gps_params):
                raise RuntimeError(
                    f"delta count {len(deltas)} != gps param count {len(self.gps_params)}"
                )
            self.gps_opt.zero_grad(set_to_none=True)
            for i, d in enumerate(deltas):
                p = self.gps_params[i]
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                p.grad.copy_(d.to(device=p.data.device, dtype=p.data.dtype, non_blocking=False))
            self.gps_opt.step()
            self.gps_opt.zero_grad(set_to_none=True)
            self.version += 1
            out = [p.detach().clone() for p in self.gps_params]
            return out, self.version


def handle_client_connection(sock: socket.socket, state: HalosGpsTcpState) -> None:
    try:
        while True:
            msg_type, payload = recv_message(sock)
            if msg_type == MSG_PING:
                send_message(sock, MSG_PONG, b"")
                continue
            if msg_type == MSG_INIT:
                data = _torch_load_bytes(payload)
                tensors = data.get("tensors")
                if not isinstance(tensors, list) or not tensors:
                    send_message(sock, MSG_ERROR, _torch_save_bytes({"error": "invalid INIT payload"}))
                    continue
                state.init_from_tensors(tensors)
                send_message(sock, MSG_ACK, _torch_save_bytes({"version": state.version}))
                continue
            if msg_type == MSG_EXCHANGE:
                data = _torch_load_bytes(payload)
                deltas = data.get("deltas")
                if not isinstance(deltas, list) or not deltas:
                    send_message(sock, MSG_ERROR, _torch_save_bytes({"error": "invalid EXCHANGE payload"}))
                    continue
                try:
                    globals_list, ver = state.apply_delta_and_return_global(deltas)
                    send_message(
                        sock,
                        MSG_RESPONSE,
                        _torch_save_bytes({"globals": globals_list, "version": ver}),
                    )
                except Exception as e:
                    logger.exception("GPS EXCHANGE failed")
                    send_message(sock, MSG_ERROR, _torch_save_bytes({"error": str(e)}))
                return
            send_message(sock, MSG_ERROR, _torch_save_bytes({"error": f"unknown msg_type {msg_type}"}))
            return
    except Exception:
        logger.exception("client handler error")
    finally:
        try:
            sock.close()
        except Exception:
            pass


class HalosRemoteGpsClient:
    """LPS-side client: INIT once, then EXCHANGE per GPS round."""

    def __init__(self, host: str, port: int, *, timeout: float = 600.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        return s

    def ping(self) -> bool:
        try:
            with self._connect() as s:
                send_message(s, MSG_PING, b"")
                t, _ = recv_message(s)
                return t == MSG_PONG
        except Exception:
            return False

    def init_server(self, tensors_cpu: Sequence[torch.Tensor]) -> None:
        payload = _torch_save_bytes(
            {"tensors": [x.detach().cpu().float().contiguous() for x in tensors_cpu]}
        )
        with self._connect() as s:
            send_message(s, MSG_INIT, payload)
            t, body = recv_message(s)
            if t == MSG_ERROR:
                err = _torch_load_bytes(body).get("error", body)
                raise RuntimeError(f"GPS INIT failed: {err}")
            if t != MSG_ACK:
                raise RuntimeError(f"GPS INIT unexpected response type {t}")

    def exchange(self, deltas_cpu: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], int]:
        payload = _torch_save_bytes(
            {"deltas": [t.detach().cpu().float().contiguous() for t in deltas_cpu]}
        )
        with self._connect() as s:
            send_message(s, MSG_EXCHANGE, payload)
            t, body = recv_message(s)
            if t == MSG_ERROR:
                err = _torch_load_bytes(body).get("error", body)
                raise RuntimeError(f"GPS EXCHANGE failed: {err}")
            if t != MSG_RESPONSE:
                raise RuntimeError(f"GPS EXCHANGE unexpected response type {t}")
            data = _torch_load_bytes(body)
            globs = data.get("globals")
            ver = int(data.get("version", 0))
            if not isinstance(globs, list):
                raise RuntimeError("GPS EXCHANGE invalid globals")
            return globs, ver
