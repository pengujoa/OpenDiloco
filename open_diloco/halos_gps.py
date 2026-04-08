"""
HALoS Global Parameter Server (GPS).

Standalone TCP server that holds the global model and processes accumulated
updates from Local Parameter Servers (LPS) using DelayedNesterov momentum.

Launch (CLI):
    python halos_gps.py --model PrimeIntellect/llama-150m-fresh \
        --port 29600 --lr 0.3 --beta 0.5 --buffer_size 2 \
        --ckpt_dir gps_ckpts --ckpt_every 32

Launch (config file — CLI args override config values):
    python halos_gps.py --config config_halos.toml
"""

import argparse
import io
import logging
import os
import socket
import struct
import threading
import time
from pathlib import Path

import torch

try:
    from halos_delayed_nesterov import DelayedNesterovOptimizer
except ImportError:
    from open_diloco.halos_delayed_nesterov import DelayedNesterovOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GPS] %(message)s")
log = logging.getLogger("halos.gps")


# ── Wire protocol (shared with LPS) ─────────────────────────────────────────

def send_tensors(sock: socket.socket, tensors: list[torch.Tensor]):
    buf = io.BytesIO()
    torch.save(tensors, buf)
    raw = buf.getvalue()
    sock.sendall(struct.pack("!Q", len(raw)) + raw)


def recv_tensors(sock: socket.socket) -> list[torch.Tensor] | None:
    hdr = _recvn(sock, 8)
    if hdr is None:
        return None
    data = _recvn(sock, struct.unpack("!Q", hdr)[0])
    if data is None:
        return None
    return torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")


def _recvn(sock: socket.socket, n: int) -> bytes | None:
    parts: list[bytes] = []
    while n > 0:
        c = sock.recv(min(n, 4_194_304))
        if not c:
            return None
        parts.append(c)
        n -= len(c)
    return b"".join(parts)


# ── GPS core ─────────────────────────────────────────────────────────────────

class GlobalParameterServer:
    def __init__(
        self,
        model_params: list[torch.Tensor],
        lr: float = 0.3,
        beta: float = 0.5,
        buffer_size: int = 2,
        c: float = 0.0,
        ckpt_dir: str | None = None,
        ckpt_every: int = 10,
    ):
        self.params = [torch.nn.Parameter(p.clone().cpu()) for p in model_params]
        self.optimizer = DelayedNesterovOptimizer(
            self.params, lr=lr, beta=beta, c=c, buffer_size=buffer_size
        )
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self.step_count = 0
        self._lps_update_counts: dict[str, int] = {}

        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_every = ckpt_every
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._try_load_latest_ckpt()

    # ── checkpoint ───────────────────────────────────────────────────────
    def _try_load_latest_ckpt(self):
        assert self.ckpt_dir is not None
        ckpts = sorted(self.ckpt_dir.glob("step_*.pt"))
        if not ckpts:
            log.info("No checkpoint found, starting fresh")
            return
        path = ckpts[-1]
        state = torch.load(path, weights_only=False, map_location="cpu")
        for p, v in zip(self.params, state["params"]):
            p.data.copy_(v)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = state["step_count"]
        log.info(f"Resumed from {path.name} (step={self.step_count})")

    def _save_ckpt(self):
        if self.ckpt_dir is None:
            return
        path = self.ckpt_dir / f"step_{self.step_count:06d}.pt"
        torch.save({
            "params": [p.data.clone() for p in self.params],
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)
        log.info(f"Checkpoint saved: {path.name}")

    # ── core update ──────────────────────────────────────────────────────
    def apply_update(self, grads: list[torch.Tensor], lps_id: str) -> list[torch.Tensor]:
        with self._lock:
            self._lps_update_counts[lps_id] = self._lps_update_counts.get(lps_id, 0) + 1

            grad_norm = sum(g.norm().item() ** 2 for g in grads) ** 0.5
            buf_pos = self.optimizer._t % self.buffer_size + 1
            will_apply_momentum = buf_pos == self.buffer_size

            self.optimizer.zero_grad()
            for p, g in zip(self.params, grads):
                p.grad = g.to(p.device)
            self.optimizer.step()
            self.step_count += 1

            log.info(
                f"step={self.step_count} | "
                f"from={lps_id} (total={self._lps_update_counts[lps_id]}) | "
                f"grad_norm={grad_norm:.4f} | "
                f"buffer={buf_pos}/{self.buffer_size} | "
                f"momentum_applied={will_apply_momentum}"
            )

            if self.step_count % self.ckpt_every == 0:
                self._save_ckpt()

            return [p.data.clone() for p in self.params]


# ── TCP server ───────────────────────────────────────────────────────────────

def serve(gps: GlobalParameterServer, host: str = "0.0.0.0", port: int = 29600):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(32)
    log.info(f"Listening on {host}:{port}")

    def _handle(conn, addr):
        lps_id = f"{addr[0]}:{addr[1]}"
        log.info(f"LPS connected: {lps_id}")
        try:
            while True:
                t0 = time.monotonic()
                updates = recv_tensors(conn)
                if updates is None:
                    break
                recv_ms = (time.monotonic() - t0) * 1000
                result = gps.apply_update(updates, lps_id)
                t1 = time.monotonic()
                send_tensors(conn, result)
                send_ms = (time.monotonic() - t1) * 1000
                log.info(f"{lps_id}: recv={recv_ms:.0f}ms, send={send_ms:.0f}ms")
        except Exception as e:
            log.error(f"{lps_id}: {e}")
        finally:
            conn.close()
            log.info(f"LPS disconnected: {lps_id}")

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=_handle, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        log.info("Shutting down")
        if gps.ckpt_dir:
            gps._save_ckpt()
            log.info("Final checkpoint saved on shutdown")
    finally:
        srv.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_config_from_toml(path: str) -> dict:
    """Read GPS-related settings from a config_halos.toml file."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    hv = cfg.get("hv", {})
    return {
        "model": cfg.get("path_model"),
        "host": "0.0.0.0",
        "port": hv.get("halos_gps_port", 29600),
        "lr": hv.get("halos_gps_lr", 0.3),
        "beta": hv.get("halos_gps_beta", 0.5),
        "buffer_size": hv.get("halos_gps_buffer_size", 2),
        "c": hv.get("halos_gps_c", 0.0),
        "ckpt_dir": hv.get("halos_gps_ckpt_dir", None),
        "ckpt_every": hv.get("halos_gps_ckpt_every", 32),
    }


def _from_pretrained_args(model_name_or_path: str) -> tuple[str, dict]:
    """로컬 디렉터리 vs Hub ID에 맞게 `from_pretrained` 인자를 맞춘다.

    Transformers는 `os.path.isdir`이 False이면 문자열을 Hub repo ID로 보내는데,
    `/mnt/...` 같은 경로는 `validate_repo_id`에서 터진다. 로컬이면 반드시 실제 디렉터리여야 한다.
    """
    expanded = os.path.expanduser(model_name_or_path)
    if os.path.isdir(expanded):
        return expanded, {"local_files_only": True}
    looks_like_filesystem_path = os.path.isabs(expanded) or model_name_or_path.startswith(
        ("./", "../", "~")
    )
    if looks_like_filesystem_path:
        raise FileNotFoundError(
            f"모델 로컬 경로가 없거나 디렉터리가 아닙니다: {expanded!r}. "
            "SFS/스토리지 마운트와 `path_model` 경로를 확인하세요."
        )
    return model_name_or_path, {}


def main():
    pa = argparse.ArgumentParser(description="HALoS Global Parameter Server")
    pa.add_argument("--config", default=None, help="Path to config_halos.toml (CLI args override config values)")
    pa.add_argument("--model", default=None, help="HuggingFace model name or path")
    pa.add_argument("--host", default=None)
    pa.add_argument("--port", type=int, default=None)
    pa.add_argument("--lr", type=float, default=None, help="GPS lr (= GLR * d_g)")
    pa.add_argument("--beta", type=float, default=None, help="GPS momentum coefficient")
    pa.add_argument("--buffer_size", type=int, default=None, help="d_g: updates buffered before momentum step")
    pa.add_argument("--c", type=float, default=None, help="Momentum activation parameter")
    pa.add_argument("--ckpt_dir", default=None, help="Directory for GPS checkpoints (enables save/resume)")
    pa.add_argument("--ckpt_every", type=int, default=None, help="Save checkpoint every N steps")
    a = pa.parse_args()

    defaults = {
        "model": None, "host": "0.0.0.0", "port": 29600,
        "lr": 0.3, "beta": 0.5, "buffer_size": 2, "c": 0.0,
        "ckpt_dir": None, "ckpt_every": 32,
    }
    if a.config:
        defaults.update(_load_config_from_toml(a.config))
        log.info(f"Loaded config from {a.config}")

    for k in defaults:
        cli_val = getattr(a, k, None)
        if cli_val is not None:
            defaults[k] = cli_val

    if defaults["model"] is None:
        pa.error("--model is required (or set path_model in config toml)")

    from transformers import AutoModelForCausalLM
    model_path, load_kw = _from_pretrained_args(defaults["model"])
    log.info(f"Loading base model: {model_path}" + (" (local_files_only)" if load_kw else ""))
    mdl = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
    params = [p.data for p in mdl.parameters()]
    del mdl
    torch.cuda.empty_cache()

    gps = GlobalParameterServer(
        params,
        lr=defaults["lr"], beta=defaults["beta"],
        buffer_size=defaults["buffer_size"], c=defaults["c"],
        ckpt_dir=defaults["ckpt_dir"], ckpt_every=defaults["ckpt_every"],
    )
    log.info(
        f"Ready: lr={defaults['lr']}, β={defaults['beta']}, d_g={defaults['buffer_size']}, c={defaults['c']}, "
        f"params={sum(p.numel() for p in gps.params):,}, "
        f"ckpt={'ON (every ' + str(defaults['ckpt_every']) + ' steps) → ' + str(defaults['ckpt_dir']) if defaults['ckpt_dir'] else 'OFF'}"
    )
    serve(gps, defaults["host"], defaults["port"])


if __name__ == "__main__":
    main()
