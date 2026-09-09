"""Heatmap-guided contact-region reward.

Pulls the 5 fingertip sites toward high-confidence contact regions of a
per-object contact_heatmap.npz (e.g. a cup handle) instead of only tracking
the (possibly mis-reconstructed) reference hand pose:

    J_contact = sum_j min_x [ ||p_j - x||^2 + alpha*(1 - H(x)) ]

    p_j   : world position of fingertip site j (thumb/index/middle/ring/pinky)
    x     : candidate object-surface point (heatmap mesh vertex, subsampled
            by confidence, see _load_heatmap_cache)
    H(x)  : contact_confidence at x, in [0, 1]
    alpha : config.contact_region_alpha

reward -= config.contact_region_rew_scale * J_contact   (lambda_c)

Design notes:

- contact_heatmap.npz's `vertices` are in the SAM-3D reconstruction's own
  scale/frame, not the compiled scene's object frame (verified: ~7x scale
  difference). But its `contact_confidence` is index-aligned 1:1 with the
  object's `{side}_visual` mesh (same vertex/face count and order, verified
  against outputs/assets/objects/cupmove/visual.obj) — so we only take
  `contact_confidence` from the npz and get vertex *positions* from the
  compiled model's own visual-mesh data (already in the right scale/frame).

- The object body has MULTIPLE mesh geoms (one high-res visual mesh + ~32
  convex collision pieces from CoACD, see generate_scene.py). We must read
  only the `{side}_visual` geom specifically — extract_body_mesh_verts (used
  elsewhere in this codebase) concatenates every mesh geom on the body and
  would break the confidence<->vertex index correspondence.

- Fingertip sites (config.hand_contact_site_ids) are ordinarily populated by
  config.py's process_config, but only when config.contact_guidance is on
  (an unrelated feature). Rather than changing that gate, this module
  resolves them itself, lazily, on first use.

Kept in its own file, isolated from mjwp.py/config.py: mjwp.py only needs one
import + one small `if` block in get_reward (same pattern as
wrist_floor_penalty.py). Nothing in config.py's control flow changes.
"""

from __future__ import annotations

import loguru
import mujoco
import numpy as np
import torch
import warp as wp

from retargeting.config import Config, build_hand_contact_site_ids
from retargeting.utils.math import mul_quat


def _find_geom_by_mesh_name(model: mujoco.MjModel, mesh_name: str) -> int:
    """Geom id whose mesh is exactly `mesh_name`, or -1 if not found."""
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    if mesh_id < 0:
        return -1
    for gi in range(model.ngeom):
        if (
            model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_dataid[gi] == mesh_id
        ):
            return gi
    return -1


def _visual_mesh_local_verts(model: mujoco.MjModel, side: str) -> np.ndarray | None:
    """Object visual-mesh vertices in the object body's local frame, or None."""
    gi = _find_geom_by_mesh_name(model, f"{side}_visual")
    if gi < 0:
        return None
    mesh_id = model.geom_dataid[gi]
    adr = int(model.mesh_vertadr[mesh_id])
    n = int(model.mesh_vertnum[mesh_id])
    v = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
    # geom pos/quat are identity for this geom in generate_scene.py's
    # `{side}_object_visual` add_geom call, but apply the transform properly
    # in case that ever changes.
    quat = np.asarray(model.geom_quat[gi], dtype=np.float64)  # wxyz
    w, x, y, z = quat
    Rg = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return v @ Rg.T + np.asarray(model.geom_pos[gi], dtype=np.float64)


def _rotate_points_by_quat(q_wxyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate `points` (..., 3) by quaternion `q_wxyz` (..., 4), broadcasting.

    Reuses mul_quat (retargeting.utils.math), already used elsewhere in this
    codebase (quat_sub) for the same MuJoCo wxyz convention.
    """
    zeros = torch.zeros(points.shape[:-1] + (1,), device=points.device, dtype=points.dtype)
    p_quat = torch.cat([zeros, points], dim=-1)
    q_conj = q_wxyz.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]
    batch_shape = torch.broadcast_shapes(q_wxyz.shape[:-1], p_quat.shape[:-1])
    q_b = q_wxyz.expand(*batch_shape, 4)
    qc_b = q_conj.expand(*batch_shape, 4)
    p_b = p_quat.expand(*batch_shape, 4)
    rotated = mul_quat(mul_quat(q_b, p_b), qc_b)
    return rotated[..., 1:]


def _load_heatmap_cache(config: Config, env) -> None:
    """Lazy, one-time setup: fingertip site ids + heatmap vertices/confidence.

    Cached on `config`, same pattern as mjwp.py's `_qpos_weight_cache`. Leaves
    config._heatmap_verts_local / config._contact_region_site_ids as None on
    any failure, so compute_contact_region_penalty degrades to a no-op
    (logged once) instead of crashing the optimizer.
    """
    if hasattr(config, "_contact_region_ready"):
        return
    config._contact_region_ready = True
    config._contact_region_site_ids = None
    config._heatmap_verts_local = None
    config._heatmap_confidence = None

    if not config.contact_heatmap_path:
        loguru.logger.warning(
            "contact_region_rew_scale > 0 but config.contact_heatmap_path is "
            "empty; contact-region reward will be a no-op."
        )
        return

    model = env.model_cpu

    site_ids = config.hand_contact_site_ids
    if not site_ids or any(s is None for s in site_ids):
        _, site_ids = build_hand_contact_site_ids(model, config.embodiment_type)
    if not site_ids or any(s is None for s in site_ids):
        loguru.logger.warning(
            "Could not resolve all 5 fingertip contact sites; "
            "contact-region reward will be a no-op."
        )
        return

    side = config.embodiment_type
    local_verts = _visual_mesh_local_verts(model, side)
    if local_verts is None:
        loguru.logger.warning(
            "No '{}_visual' mesh geom found; contact-region reward will be a no-op.",
            side,
        )
        return

    data = np.load(config.contact_heatmap_path)
    confidence = data["contact_confidence"].astype(np.float64)
    if confidence.shape[0] != local_verts.shape[0]:
        loguru.logger.warning(
            "contact_heatmap.npz vertex count ({}) != {}_visual mesh vertex "
            "count ({}); contact-region reward will be a no-op.",
            confidence.shape[0], side, local_verts.shape[0],
        )
        return

    # Subsample to the highest-confidence points. Low-confidence vertices
    # rarely win the min() anyway (their alpha*(1-H) term is large), so this
    # barely changes the result while bounding the per-step pairwise-distance
    # cost (worlds x 5 fingers x V points, every reward call).
    max_pts = config.contact_region_max_points
    if max_pts > 0 and confidence.shape[0] > max_pts:
        keep = np.argpartition(-confidence, max_pts)[:max_pts]
    else:
        keep = np.arange(confidence.shape[0])

    device = config.device
    config._contact_region_site_ids = [int(s) for s in site_ids]
    config._heatmap_verts_local = torch.tensor(
        local_verts[keep], dtype=torch.float32, device=device
    )
    config._heatmap_confidence = torch.tensor(
        confidence[keep], dtype=torch.float32, device=device
    )
    loguru.logger.info(
        "Contact-region reward: loaded {} heatmap points (of {}) from {}",
        config._heatmap_verts_local.shape[0], confidence.shape[0],
        config.contact_heatmap_path,
    )


def compute_contact_region_penalty(
    config: Config, env, qpos_sim: torch.Tensor
) -> torch.Tensor:
    """Per-world penalty: config.contact_region_rew_scale * J_contact.

    Returns zeros (no-op) if setup failed (missing sites / heatmap / mesh
    mismatch) — see _load_heatmap_cache.
    """
    _load_heatmap_cache(config, env)
    if config._heatmap_verts_local is None or config._contact_region_site_ids is None:
        return torch.zeros(qpos_sim.shape[0], device=qpos_sim.device)

    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (worlds, nsite, 3)
    fingertips = site_xpos[:, config._contact_region_site_ids]  # (worlds, 5, 3)

    obj_pos = qpos_sim[:, -7:-4]  # (worlds, 3)
    obj_quat = qpos_sim[:, -4:]  # (worlds, 4) wxyz

    verts_local = config._heatmap_verts_local  # (V, 3)
    verts_world = (
        _rotate_points_by_quat(obj_quat[:, None, :], verts_local[None, :, :])
        + obj_pos[:, None, :]
    )  # (worlds, V, 3)

    diff = fingertips[:, :, None, :] - verts_world[:, None, :, :]  # (worlds, 5, V, 3)
    dist_sq = (diff ** 2).sum(-1)  # (worlds, 5, V)
    confidence = config._heatmap_confidence[None, None, :]  # (1, 1, V)
    cost = dist_sq + config.contact_region_alpha * (1.0 - confidence)  # (worlds, 5, V)
    per_finger_min = cost.min(dim=-1).values  # (worlds, 5)
    total = per_finger_min.sum(dim=-1)  # (worlds,)
    return config.contact_region_rew_scale * total
