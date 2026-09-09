"""One-off: export a GLB showing the centroid-based warmup retreat direction.

Recomputes the exact same hand/object centroid + retreat vector that
warmup_analytical_init (retargeting/utils/mjwp.py) computes, using the same
helper functions, then builds a single self-contained 3D scene:

  - object mesh (reference pose)
  - hand mesh BEFORE warmup (semi-transparent orange)
  - hand mesh AFTER warmup (solid green)
  - small spheres at the hand/object centroids
  - a cylinder from the pre-warmup hand centroid to the post-warmup one,
    i.e. the actual retreat vector applied

Run from retargeting/ in the `retargeting` conda env:
    python scripts_scratch_viz_centroid_direction.py --run-dir outputs/sharpa/left/cupmove/0
Open the resulting .glb in any glTF viewer (e.g. https://gltf-viewer.donmccurdy.com/,
Blender, or VS Code's glTF preview) and orbit with the mouse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from retargeting.utils.in_hand import body_has_mesh_geom, extract_body_mesh_verts
from retargeting.utils.mjwp import _object_mesh_body_id
from retargeting.utils.viser_viewer import _mujoco_mesh_to_trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/sharpa/left/cupmove/0")
    p.add_argument("--side", default="left")
    p.add_argument("--clearance", type=float, default=None, help="Override warmup_min_clearance (meters); default reads config.yaml.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _read_clearance(run_dir: Path) -> float:
    cfg_path = run_dir / "config.yaml"
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(cfg_path))
    return float(cfg.get("warmup_min_clearance", 0.1))


def _world_mesh(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, rgba) -> list[trimesh.Trimesh]:
    """All mesh geoms on body_id, posed to world, as separate colored trimeshes."""
    out = []
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        m = _mujoco_mesh_to_trimesh(model, gi)
        if m is None:
            continue
        # geom-local -> world: apply geom xpos/xmat directly (post-FK, already
        # composed body+geom transform), matching data.geom_xpos/geom_xmat.
        R = data.geom_xmat[gi].reshape(3, 3)
        t = data.geom_xpos[gi]
        m.apply_transform(np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]))
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.tile(rgba, (len(m.vertices), 1)))
        out.append(m)
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))

    kin = np.load(str(run_dir / "trajectory_kinematic.npz"))
    qpos_ref_0 = kin["qpos"][0].astype(np.float64)

    clearance = args.clearance if args.clearance is not None else _read_clearance(run_dir)

    # --- FK at the reference (pre-warmup) pose, exactly like warmup_analytical_init ---
    fk_pre = mujoco.MjData(model)
    fk_pre.qpos[:] = qpos_ref_0
    mujoco.mj_kinematics(model, fk_pre)

    side = args.side
    obj_body_id = _object_mesh_body_id(model, side)
    hand_parts = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name.startswith(f"{side}_") or bid == obj_body_id:
            continue
        if not body_has_mesh_geom(model, bid):
            continue
        hand_parts.append(extract_body_mesh_verts(model, bid, data=fk_pre, apply_geom_xform=True))
    hand_pts = np.concatenate(hand_parts, axis=0)
    obj_verts_w = extract_body_mesh_verts(model, obj_body_id, data=fk_pre, apply_geom_xform=True)

    # --- Same centroid-direction computation as warmup_analytical_init ---
    hand_centroid = hand_pts.mean(axis=0)
    obj_centroid = obj_verts_w.mean(axis=0)
    away = hand_centroid - obj_centroid
    away_norm = float(np.linalg.norm(away))
    n_world = -away / away_norm
    t_chosen = clearance
    offset = t_chosen * n_world
    print(f"hand_centroid={hand_centroid}\nobj_centroid={obj_centroid}\n"
          f"retreat_dir(away from object)={-n_world}\nt_chosen={t_chosen:.3f}m")

    # --- Wrist pos_idx: single-hand embodiment -> qpos[0:3] ---
    qpos_post = qpos_ref_0.copy()
    qpos_post[0:3] = qpos_post[0:3] - offset
    floor_margin = 0.01
    qpos_post[2] = max(qpos_post[2], 0.0 + floor_margin)

    fk_post = mujoco.MjData(model)
    fk_post.qpos[:] = qpos_post
    mujoco.mj_kinematics(model, fk_post)

    hand_centroid_post = hand_centroid - offset  # matches the rigid wrist translation

    # --- Build the scene ---
    scene = trimesh.Scene()

    ORANGE = np.array([255, 140, 40, 140], dtype=np.uint8)   # pre-warmup hand, translucent
    GREEN = np.array([60, 200, 90, 255], dtype=np.uint8)     # post-warmup hand, solid
    GRAY = np.array([160, 160, 160, 255], dtype=np.uint8)    # object
    RED_SPHERE = np.array([220, 40, 40, 255], dtype=np.uint8)
    BLUE_SPHERE = np.array([40, 90, 220, 255], dtype=np.uint8)
    ARROW = np.array([250, 200, 30, 255], dtype=np.uint8)

    for m in _world_mesh(model, fk_pre, obj_body_id, GRAY):
        scene.add_geometry(m)
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name.startswith(f"{side}_") or bid == obj_body_id:
            continue
        if not body_has_mesh_geom(model, bid):
            continue
        for m in _world_mesh(model, fk_pre, bid, ORANGE):
            scene.add_geometry(m)
        for m in _world_mesh(model, fk_post, bid, GREEN):
            scene.add_geometry(m)

    def add_sphere(center, rgba, radius=0.006):
        s = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        s.apply_translation(center)
        s.visual = trimesh.visual.ColorVisuals(mesh=s, vertex_colors=np.tile(rgba, (len(s.vertices), 1)))
        scene.add_geometry(s)

    def add_arrow(p0, p1, rgba, radius=0.003):
        vec = np.asarray(p1) - np.asarray(p0)
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        # cylinder is created along +z centered at origin; rotate +z -> vec, then move to midpoint
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, vec / length)
        angle = np.arccos(np.clip(np.dot(z, vec / length), -1.0, 1.0))
        if np.linalg.norm(axis) > 1e-8:
            R = trimesh.transformations.rotation_matrix(angle, axis)
        else:
            R = np.eye(4) if angle < 1e-6 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        cyl.apply_transform(R)
        cyl.apply_translation((np.asarray(p0) + np.asarray(p1)) / 2.0)
        cyl.visual = trimesh.visual.ColorVisuals(mesh=cyl, vertex_colors=np.tile(rgba, (len(cyl.vertices), 1)))
        scene.add_geometry(cyl)
        head = trimesh.creation.cone(radius=radius * 2.2, height=radius * 6, sections=16)
        head.apply_transform(R)
        head.apply_translation(np.asarray(p1))
        head.visual = trimesh.visual.ColorVisuals(mesh=head, vertex_colors=np.tile(rgba, (len(head.vertices), 1)))
        scene.add_geometry(head)

    add_sphere(hand_centroid, RED_SPHERE)
    add_sphere(obj_centroid, BLUE_SPHERE)
    add_sphere(hand_centroid_post, RED_SPHERE)
    add_arrow(hand_centroid, hand_centroid_post, ARROW)

    out_path = Path(args.out) if args.out else run_dir / "warmup_fix_check" / "centroid_direction_viz.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
