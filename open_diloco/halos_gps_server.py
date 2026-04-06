"""
Standalone HALoS GPS TCP server (separate process).

Example:
  python -m open_diloco.halos_gps_server --host 0.0.0.0 --port 9100 \\
    --gps-lr 0.3 --gps-beta 0.5 --gps-buffer-size 2

Training processes must bootstrap GPS with the same initial weights as LPS
(the first LPS messenger typically sends MSG_INIT once via halos_lps_gps remote mode).
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading

try:
    from halos_gps_tcp import HalosGpsTcpState, handle_client_connection
except ImportError:
    from open_diloco.halos_gps_tcp import HalosGpsTcpState, handle_client_connection  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="HALoS dedicated GPS (TCP)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--gps-lr", type=float, default=0.3)
    p.add_argument("--gps-beta", type=float, default=0.5)
    p.add_argument("--gps-buffer-size", type=int, default=2)
    p.add_argument("--gps-c", type=float, default=0.0)
    args = p.parse_args()

    state = HalosGpsTcpState(
        gps_lr=args.gps_lr,
        gps_beta=args.gps_beta,
        gps_buffer_size=args.gps_buffer_size,
        gps_c=args.gps_c,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(128)
    logger.info(
        "HALoS GPS listening on %s:%s (lr=%s beta=%s buf=%s c=%s). "
        "Wait for clients to send INIT before EXCHANGE.",
        args.host,
        args.port,
        args.gps_lr,
        args.gps_beta,
        args.gps_buffer_size,
        args.gps_c,
    )

    while True:
        conn, addr = srv.accept()
        logger.info("GPS client connected from %s", addr)
        t = threading.Thread(
            target=handle_client_connection,
            args=(conn, state),
            daemon=True,
        )
        t.start()


if __name__ == "__main__":
    main()
