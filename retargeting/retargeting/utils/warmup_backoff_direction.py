"""Heatmap-based warmup backoff direction (opt-in).

With ``config.warmup_backoff_mode == "heatmap"``, warmup_analytical_init
translates the hand along the *handle direction* — from the object centroid
toward the center of the contact heatmap's high-confidence region — instead
of along the hand-centroid -> object-centroid direction. The hand then starts
warmup on the grasp-target side of the object (visualized/approved via
viz_backoff_compare.py's blue hand, 2026-08-26).

Requires ``config.contact_heatmap_path`` to point to an *object-order*
heatmap npz (``contact_confidence`` index-aligned with the compiled
``{side}_visual`` mesh — see reconstruction/fix_heatmap_vertex_order.py; the
raw manual_contact_heatmap.py output is in SAM-3D vertex order and must be
remapped first). Falls back to the centroid direction (returns None, logged)
on any failure, so the opt-in mode can never crash the init.
"""

from __future__ import annotations

import loguru
import mujoco
import numpy as np

from retargeting.config import Config
from retargeting.utils.contact_region_reward import _find_geom_by_mesh_name

# Vertices with confidence above this count as the grasp-target ("handle")
# region whose center defines the backoff direction. Matches the threshold
# used in viz_backoff_compare.py.
_CONF_THRESH = 0.5


def heatmap_backoff_direction(
    config: Config,
    model: mujoco.MjModel,
    fk_data: mujoco.MjData,
    side: str,
    obj_centroid: np.ndarray,
) -> np.ndarray | None:
    """Unit vector (world) from ``obj_centroid`` toward the heatmap's
    high-confidence region center, or None if it cannot be computed."""
    if not config.contact_heatmap_path:
        loguru.logger.warning(
            "warmup_backoff_mode='heatmap' but contact_heatmap_path is empty; "
            "falling back to centroid direction."
        )
        return None

    gi = _find_geom_by_mesh_name(model, f"{side}_visual")
    if gi < 0:
        loguru.logger.warning(
            "warmup_backoff_mode='heatmap': no '{}_visual' mesh geom; "
            "falling back to centroid direction.", side,
        )
        return None

    data = np.load(config.contact_heatmap_path)
    conf = np.asarray(data["contact_confidence"], dtype=np.float64)
    mesh_id = int(model.geom_dataid[gi])
    adr = int(model.mesh_vertadr[mesh_id])
    n = int(model.mesh_vertnum[mesh_id])
    if conf.shape[0] != n:
        loguru.logger.warning(
            "warmup_backoff_mode='heatmap': heatmap vertex count ({}) != "
            "{}_visual mesh vertex count ({}); falling back to centroid "
            "direction.", conf.shape[0], side, n,
        )
        return None

    high = conf > _CONF_THRESH
    if not high.any():
        loguru.logger.warning(
            "warmup_backoff_mode='heatmap': no vertices above confidence {}; "
            "falling back to centroid direction.", _CONF_THRESH,
        )
        return None

    verts_local = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
    R = fk_data.geom_xmat[gi].reshape(3, 3)
    t = fk_data.geom_xpos[gi]
    handle_center = verts_local[high] @ R.T + t
    direction = handle_center.mean(axis=0) - np.asarray(obj_centroid, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    direction = direction / norm
    loguru.logger.info(
        "Warmup ({}): heatmap backoff direction {} (handle-region center of "
        "{} verts).", side, np.round(direction, 3), int(high.sum()),
    )
    return direction
