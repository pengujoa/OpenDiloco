#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, socket, os
from typing import Dict
from hivemind.dht import DHT
from hivemind.utils import get_dht_time

def unwrap(v):
    return getattr(v, "value", v)

def read_speeds(dht: DHT, key: str) -> Dict[str, float]:
    res = dht.get(key, latest=True)
    root = unwrap(res) if res else None
    speeds: Dict[str, float] = {}
    if isinstance(root, dict):
        for k, v in root.items():
            p = unwrap(v)
            if isinstance(p, dict):
                if "steps_per_sec" in p and isinstance(p["steps_per_sec"], (int, float)):
                    speeds[k] = float(p["steps_per_sec"])
                elif "v" in p and isinstance(p["v"], (int, float)):
                    speeds[k] = float(p["v"])
                elif "step_time_s" in p and isinstance(p["step_time_s"], (int, float)) and p["step_time_s"] > 0:
                    speeds[k] = 1.0 / float(p["step_time_s"])
    return speeds

def main():
    ap = argparse.ArgumentParser(description="Compute per-worker local steps (H) and publish them back to DHT")
    ap.add_argument("--initial-peer", action="append", required=True, help="libp2p multiaddr")
    ap.add_argument("--run-id", default="OpenDiLoCo", help="RUN_ID used for DHT keys")
    ap.add_argument("--speed-key", default="speed", help="suffix for speed key (default: speed)")
    ap.add_argument("--hfast", type=int, default=128, help="H_fast (fastest worker's local steps)")
    ap.add_argument("--ttl", type=float, default=60.0, help="expiration time in seconds for published H values")
    args = ap.parse_args()

    dht = DHT(initial_peers=args.initial_peer, start=True, client_mode=False)

    # 1. Read speeds
    key_speed = f"{args.run_id}:{args.speed_key}"
    speeds = read_speeds(dht, key_speed)
    if not speeds:
        print(f"[error] no usable speeds at '{key_speed}'. Is the publisher running?")
        return 2

    # 2. Compute H values
    vmax = max(speeds.values())
    print(f"v_max={vmax:.2f}  H_fast={args.hfast}")

    # 3. Publish results to DHT
    key_h = f"{args.run_id}:H"
    now = get_dht_time()
    exp = now + args.ttl

    for k in sorted(speeds, key=speeds.get, reverse=True):
        rel = speeds[k] / vmax
        H = int(math.floor(rel * args.hfast))

        payload = {"H": H, "v": speeds[k], "rel": rel, "ts": now}
        ok = dht.store(key=key_h, subkey=k, value=payload, expiration_time=exp)

        print(f"{k:50s} v={speeds[k]:8.2f}  rel={rel:6.3f}  H={H:3d} ({'ok' if ok else 'fail'})")

    print(f"[done] Published H values under key '{key_h}'")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
