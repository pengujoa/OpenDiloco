"""
2-tier communication: regional (LPS) via torch.distributed, global (GPS) via Hivemind allreduce.

Each region groups Hivemind `world_rank` indices (same node or same geo region).
Messengers (local_rank==0) run `all_reduce(SUM)/region_size` on pseudo-grad tensors
before decentralized averaging, so hivemind sees region means — global mean matches
hierarchical averaging for any partition sizes.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch.distributed as dist

logger = logging.getLogger(__name__)


def validate_halos_regions(regions: List[List[int]], galaxy_size: int) -> None:
    flat = [x for g in regions for x in g]
    if len(flat) != len(set(flat)):
        raise ValueError(f"halos_regions must not duplicate world_rank: {regions}")
    if sorted(flat) != list(range(galaxy_size)):
        raise ValueError(
            f"halos_regions must partition 0..galaxy_size-1 exactly (galaxy_size={galaxy_size}), got {regions}"
        )


def default_messenger_global_ranks(world_size: int, local_world_size: int, galaxy_size: int) -> List[int]:
    if local_world_size < 1:
        local_world_size = 1
    ranks = [i * local_world_size for i in range(galaxy_size)]
    if not ranks or ranks[-1] >= world_size:
        raise ValueError(
            f"Cannot infer messenger ranks: galaxy_size={galaxy_size}, local_world_size={local_world_size}, "
            f"world_size={world_size}. Set [hv] halos_messenger_global_ranks to length galaxy_size."
        )
    return ranks


def build_halos_regional_process_groups(
    *,
    torch_rank: int,
    world_size: int,
    local_rank: int,
    galaxy_size: int,
    hv_world_rank: int,
    regions: List[List[int]],
    messenger_global_ranks: Optional[List[int]],
    local_world_size: int,
) -> Tuple[Optional[dist.ProcessGroup], int]:
    """
    All processes must call this after init_process_group, in the same order.

    Returns (regional_process_group, region_size) for DiLoCo messenger processes that
    participate in a multi-peer region; otherwise (None, 1).
    """
    if not dist.is_initialized():
        return None, 1

    if not regions:
        raise ValueError(
            "halos_hierarchical_ps requires [hv].halos_regions as a partition of galaxy world ranks, "
            "e.g. [[0, 1], [2, 3]] for galaxy_size = 4."
        )

    validate_halos_regions(regions, galaxy_size)

    if messenger_global_ranks is None:
        messenger_global_ranks = default_messenger_global_ranks(world_size, local_world_size, galaxy_size)
    elif len(messenger_global_ranks) != galaxy_size:
        raise ValueError(
            f"halos_messenger_global_ranks must have length galaxy_size={galaxy_size}, "
            f"got {len(messenger_global_ranks)}"
        )

    wr_to_torch = {wr: messenger_global_ranks[wr] for wr in range(galaxy_size)}

    my_group: Optional[dist.ProcessGroup] = None
    my_size = 1

    for region in regions:
        member_ranks = sorted(wr_to_torch[wr] for wr in region)
        pg = dist.new_group(ranks=member_ranks)
        if torch_rank in member_ranks:
            my_group = pg
            my_size = len(member_ranks)

    if local_rank != 0:
        return None, 1

    if torch_rank not in messenger_global_ranks:
        return None, 1

    if my_group is not None and my_size > 1:
        logger.info(
            "[HALoS hierarchical] LPS (torch): messenger RANK=%s hv_world_rank=%s in regional group size=%s %s",
            torch_rank,
            hv_world_rank,
            my_size,
            next((r for r in regions if hv_world_rank in r), None),
        )
    return my_group, my_size
