"""Distance gate for the warmup analytical backoff (opt-in).

The analytical backoff (warmup_analytical_init in mjwp.py) exists to rescue
demos whose *reference* already starts with the hand resting on / touching the
object: without it the optimizer starts inside the object and warmup has no
room to find a pre-grasp. Demos that already start with the hand well clear of
the object don't have that problem, and backing them off another
``warmup_min_clearance`` meters only moves the start away from the reference
for no reason.

With ``config.warmup_backoff_trigger_dist > 0`` the backoff is applied only
when the robot hand's mesh comes within that distance of the object mesh at
the reference pose; otherwise the hand is left exactly where the reference
puts it (t = 0, i.e. the original do-as-i-do behaviour).

Measured at the reference pose (2026-08-26), with a 0.10 m trigger:
    cupmove      0.0419 m  -> gate fires   (backoff applied, as before)
    dexycb_side  0.2599 m  -> gate skipped (start pose == reference)

Note this is the robot hand's own mesh distance, the same geometry the backoff
itself uses — not resolve_pedestal.py's MANO-surface ``in_hand`` check, which
measures a different (smaller) hand and so reports different numbers for the
same frame.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from retargeting.config import Config


def should_apply_backoff(
    config: Config,
    hand_pts: np.ndarray,
    obj_verts_w: np.ndarray,
) -> tuple[bool, float]:
    """``(apply, ref_min_dist)`` for one hand at the reference pose.

    ``hand_pts`` / ``obj_verts_w`` are world-frame vertices (already
    subsampled by the caller). Returns ``ref_min_dist = nan`` when no gate is
    configured, since the distance is then never computed.
    """
    trigger = float(config.warmup_backoff_trigger_dist)
    if trigger <= 0.0:
        return True, float("nan")
    if hand_pts.shape[0] == 0 or obj_verts_w.shape[0] == 0:
        return True, float("nan")
    ref_min_dist = float(cKDTree(obj_verts_w).query(hand_pts)[0].min())
    return ref_min_dist <= trigger, ref_min_dist
