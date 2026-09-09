"""Wrist-floor penalty: discourage the wrist from dipping below the floor.

do_as_i_do runs with hand_floor_collision disabled (see launch.py), so there
is no physical floor to stop the wrist during a control-tracking transient —
observed concretely around the warmup end (weld released, gravity restored),
where the wrist dips a few cm below z=0 and recovers within ~0.3s even though
neither the interpolated warmup reference nor the raw reference ever goes
near the floor. This adds a soft reward-side deterrent, the same
clamp-squared pattern as mjwp._get_object_z + the drop_penalty block, so the
sampler itself downweights control candidates that dip below the floor
instead of relying on (absent) physical collision to stop them.

Kept in its own file, isolated from mjwp.py/config.py, so the wiring is easy
to spot and revert: two field declarations in config.py, one import + one
`if` block in mjwp.get_reward.
"""

from __future__ import annotations

import torch

from retargeting.config import Config


def get_wrist_z(config: Config, qpos_sim: torch.Tensor) -> torch.Tensor:
    """Per-world wrist z (one column per hand): (num_worlds, num_hands)."""
    if config.embodiment_type == "bimanual":
        robot_nq = config.nq - config.nq_obj
        half = robot_nq // 2
        return torch.stack([qpos_sim[:, 2], qpos_sim[:, half + 2]], dim=1)
    elif config.embodiment_type in ("right", "left"):
        return qpos_sim[:, 2].unsqueeze(1)
    else:
        return torch.zeros(qpos_sim.shape[0], 1, device=qpos_sim.device)


def compute_wrist_floor_penalty(
    config: Config, qpos_sim: torch.Tensor
) -> torch.Tensor:
    """Squared-clamp penalty, per world, for wrist z below wrist_floor_margin.

    Same shape as mjwp's drop_penalty: 0 if the wrist stays at/above the
    margin, growing with the square of how far below it dips.
    """
    wrist_z = get_wrist_z(config, qpos_sim)
    below = torch.clamp(config.wrist_floor_margin - wrist_z, min=0.0)
    return config.wrist_floor_penalty_scale * (below * below).sum(dim=-1)
