"""Keep the warmup reference-interpolation *target* as separated from the
object as the analytically-placed *start*, instead of collapsing back to the
original (near-touching) reference pose exactly when warmup ends.

optimize_physics.py's warmup reference interpolation blends the wrist from
`selected_qpos` (wherever warmup_analytical_init just placed it, backed off
from the object) toward `base_target_qpos` == `qpos_ref[warmup_ref_base_interp_steps]`.
Because io.py prepends `warmup_steps` copies of frame 0 to the reference
(`qpos_ref_interp[:1].expand(n, -1)`), and warmup_ref_base_interp_steps ==
warmup_steps for do_as_i_do, that target is *exactly* the raw, un-backed-off
frame-0 pose again. So by construction the interpolated target -- and
therefore the wrist -- ends up right back next to the object exactly when
weld releases and free dynamics begin, undoing whatever separation
warmup_analytical_init bought at t=0 (confirmed empirically: cupmove_noinit,
which skips the backoff entirely, and a 30cm backoff both converge to the
same near-touching state by the time warmup ends).

This shifts the interpolation target's wrist position(s) by the same offset
already applied to reach `selected_qpos`/`selected_ctrl` from
`qpos_ref[0]`/`ctrl_ref[0]`, so the wrist stays separated through the whole
warmup window, right up to (and including) the moment it ends. Orientation,
fingers, and object dims are untouched -- only wrist *position* is held
back; the actual approach-and-grasp motion happens afterward, driven by real
reward-based tracking of the true reference once free dynamics begin.
"""

from __future__ import annotations

import numpy as np

from retargeting.config import Config


def _wrist_pos_slices(config: Config) -> list[tuple[int, int]]:
    """(start, end) index pairs for each hand's wrist position (3 DOF)."""
    if config.embodiment_type == "bimanual":
        robot_nq = config.nq - config.nq_obj
        half = robot_nq // 2
        return [(0, 3), (half, half + 3)]
    elif config.embodiment_type in ("right", "left"):
        return [(0, 3)]
    return []


def shift_target_position(
    config: Config,
    target: np.ndarray,
    ref_0: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Shift `target`'s wrist position slice(s) by (ref_0 - selected) there.

    `target`, `ref_0`, `selected` must share the same DOF indexing (all
    qpos-space, or all ctrl-space) — this only touches the wrist position
    slice(s); everything else in `target` is left as-is.
    """
    out = target.copy()
    for start, end in _wrist_pos_slices(config):
        offset = ref_0[start:end] - selected[start:end]
        out[start:end] = out[start:end] - offset
    return out
