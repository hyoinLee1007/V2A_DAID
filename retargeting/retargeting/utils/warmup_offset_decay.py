"""Give the hand an approach to follow instead of a teleport to chase.

``warmup_target_offset.py`` holds the wrist backed off from the object for the
whole warmup, so the analytical init's separation is not undone the moment
free dynamics begin. It is applied to ``qpos_ref[:warmup_steps]`` and simply
not applied afterwards, and that boundary is a cliff: measured on cupmove, the
reference wrist moves 0.000 mm for steps 0-598 and then **191.4 mm in the
single step** from 599 to 600.

The hand travels about 1.6 mm per step, so it needs ~120 steps to cover that.
It ends warmup 152 mm behind the reference and is still 44 mm behind on
average afterwards — and the grasp happens inside that catch-up, with the wrist
several centimetres from where the reference says it should be. Same finger
angles at the wrong wrist position is how a pad contact becomes a back-of-
finger contact.

That offset's own docstring says "the actual approach-and-grasp motion happens
afterward, driven by real reward-based tracking of the true reference once free
dynamics begin". This makes that true: rather than dropping the offset at the
boundary, it is faded out over the first ``decay_steps`` of free dynamics, so
the reference itself contains the approach and the hand can track it.

Removing the offset *inside* warmup instead would put the wrist back against
the object exactly when the weld releases, which is the failure
``warmup_target_offset.py`` exists to prevent.
"""

from __future__ import annotations

import numpy as np

from retargeting.config import Config
from retargeting.utils.warmup_target_offset import _wrist_pos_slices


def decay_profile(n: int) -> np.ndarray:
    """Weights 1 -> 0 over ``n`` steps, ending exactly at 0.

    Smoothstep rather than linear: a linear fade leaves a velocity step at both
    ends (the reference is stationary before the boundary and moving at the
    fade rate just after it), and the sampler has to absorb that discontinuity
    at the exact moment contact dynamics turn on.
    """
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n, dtype=np.float64) / float(n)
    return 1.0 - (t * t * (3.0 - 2.0 * t))


def apply_offset_decay(config: Config, qpos_ref, ctrl_ref, ref_0_qpos,
                       selected_qpos, ref_0_ctrl, selected_ctrl) -> int:
    """Fade the warmup wrist offset out over the first steps of free dynamics.

    Mutates ``qpos_ref``/``ctrl_ref`` in place from ``warmup_steps`` onward.
    Returns the number of steps faded (0 when disabled or out of range).

    The offset is the same quantity ``shift_target_position`` subtracts, so at
    step ``warmup_steps`` this reproduces exactly the pose the warmup ended on
    and the boundary becomes continuous.
    """
    import torch

    n = int(config.warmup_offset_decay_steps)
    start = int(config.warmup_steps)
    if n <= 0 or start <= 0 or start >= qpos_ref.shape[0]:
        return 0
    n = min(n, qpos_ref.shape[0] - start)
    w = decay_profile(n)

    for arr, ref0, sel in ((qpos_ref, ref_0_qpos, selected_qpos),
                           (ctrl_ref, ref_0_ctrl, selected_ctrl)):
        if arr is None or ref0 is None or sel is None:
            continue
        for lo, hi in _wrist_pos_slices(config):
            if hi > arr.shape[1] or hi > len(ref0) or hi > len(sel):
                continue
            offset = np.asarray(ref0[lo:hi], np.float64) - np.asarray(sel[lo:hi], np.float64)
            shift = torch.from_numpy(np.outer(w, offset)).to(arr)
            arr[start:start + n, lo:hi] = arr[start:start + n, lo:hi] - shift
    return n
