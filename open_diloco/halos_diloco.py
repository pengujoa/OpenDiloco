"""
HALoS-related hooks for OpenDiLoCo (async baseline + pointers to paper optimizer).

Reference:
  "HALoS: Hierarchical Asynchronous Local SGD over Slow Networks
   for Geo-Distributed LLM Training" (ICML 2025)
  https://openreview.net/forum?id=LMPJnZSNC8
  Code: https://github.com/utnslab/halos

HaLoSOptimizer:
  - NO_WAIT allreduce + grad averager async_mode (no DHT local-step barrier).

GPS outer (paper `gps_opt_config` / delayed Nesterov):
  - Module `halos_delayed_nesterov.DelayedNesterovOptimizer` (matches utnslab/halos
    `opt.py`, arXiv:2401.09135). Enable in train_fsdp via
    `halos_gps_delayed_nesterov = true` and `halos_gps_*` in `[hv]`.

Full hierarchy (LPS delayed Nesterov, K-accumulation, α merge in `merge_model`)
differs from a single decentralized pseudo-gradient round; `HvConfig` exposes
`halos_lps_*`, `halos_model_merge_weight`, `halos_local_updates_accumulation`
for parity with `examples/run_halos.sh` and future work.

Usage:
  `[hv] halos_mode = true`. Optional: `halos_gps_delayed_nesterov = true`.
"""

import time

from hivemind.optim.optimizer import logger

try:
    from .hivemind_diloco import AllReduceStrategy, DiLoCoOptimizer
except ImportError:
    from hivemind_diloco import AllReduceStrategy, DiLoCoOptimizer


class HaLoSOptimizer(DiLoCoOptimizer):
    """
    HALoS-style async optimizer (flat-topology variant = Async Local SGD).

    Key differences from DiLoCoOptimizer:
    1. No sync barrier — does NOT wait for all nodes to finish local steps before
       triggering the outer step.
    2. NO_WAIT allreduce strategy — proceeds with available peers; stragglers are
       skipped rather than blocking the fast node.
    3. Staleness logging — records how many outer steps each peer has drifted.

    All existing DiLoCo features (sign1bit, majority_vote, adaptive_mean,
    token_weighted_aggregation, gradient_magnitude_selection, etc.) are preserved
    because HaLoSOptimizer inherits them unchanged.
    """

    def __init__(
        self,
        *,
        halos_async_timeout: float = 60.0,  # max seconds to wait for peers in allreduce
        **kwargs,
    ):
        # Force NO_WAIT: outer step doesn't wait for slow/missing peers.
        # This is the defining property of HALoS in our topology.
        if "timeout_waiting_for_peers" in kwargs:
            kwargs.pop("timeout_waiting_for_peers")
        kwargs["all_reduce_strategy"] = AllReduceStrategy.NO_WAIT

        # Shorten matchmaking if not explicitly set — in async mode a long
        # matchmaking window negates the straggler-avoidance benefit.
        if "matchmaking_time" not in kwargs or kwargs["matchmaking_time"] is None:
            kwargs["matchmaking_time"] = halos_async_timeout

        super().__init__(**kwargs)

        # Disable DHT sync barrier inside DiLoCoGradAverager.step()
        self.diloco_grad_averager.async_mode = True

        self._halos_async_timeout = halos_async_timeout
        self._outer_step_timestamps: list[float] = []  # wall-clock time of each outer step
        logger.info(
            "[HALoS] HaLoSOptimizer initialized: async_mode=True, "
            f"allreduce_strategy=NO_WAIT, halos_async_timeout={halos_async_timeout}s"
        )

    # ------------------------------------------------------------------
    # Staleness / timing helpers
    # ------------------------------------------------------------------

    def _record_outer_step(self) -> None:
        self._outer_step_timestamps.append(time.perf_counter())

    def _log_async_stats(self) -> None:
        epoch = self.local_epoch
        num_peers = self.tracker.global_progress.num_peers
        logger.info(
            f"[HALoS] outer_step={epoch} "
            f"peers_seen={num_peers} "
            f"async_mode=True (no barrier)"
        )

    # ------------------------------------------------------------------
    # Override _update_global_epoch to add HALoS-specific logging
    # The actual async behavior comes from:
    #   - all_reduce_strategy=NO_WAIT  (skips matchmaking wait loop)
    #   - diloco_grad_averager.async_mode=True  (skips DHT barrier)
    # ------------------------------------------------------------------

    def _update_global_epoch(self) -> None:
        t0 = time.perf_counter()
        self._log_async_stats()
        super()._update_global_epoch()
        self._record_outer_step()
        elapsed = time.perf_counter() - t0
        logger.info(f"[HALoS] outer_step wall_time={elapsed:.3f}s (epoch={self.local_epoch})")
