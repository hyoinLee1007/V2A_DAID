"""One-off: compare warmup backoff directions in a single GLB.

Question this answers (see conversation 2026-08-26): would backing the hand
off along the *handle direction* (heatmap high-confidence center - object
center) start the optimization in a better basin than the current
centroid-based direction (hand center - object center)? The demo wrist path
is drawn too, because the reference approach is *curved* toward the handle —
so the terminal grasp posture may be dictated by the reference regardless of
where the backoff puts the hand at t=0.

Scene contents:
  - object mesh, vertex-colored by contact_heatmap confidence
    (gray -> red = handle)
  - hand at reference frame 0            : translucent ORANGE
  - hand backed off, centroid direction  : GREEN  (current method)
  - hand backed off, handle direction    : BLUE   (proposed method)
  - demo wrist path over all frames      : small spheres, light -> dark = time
  - YELLOW arrow  : centroid retreat vector
  - MAGENTA arrow : handle-direction retreat vector

Run from retargeting/ in the `retargeting` conda env:
    python viz_backoff_compare.py --run-dir outputs/sharpa/left/cupmove/0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from retargeting.utils.contact_region_reward import _find_geom_by_mesh_name
from retargeting.utils.in_hand import body_has_mesh_geom, extract_body_mesh_verts
from retargeting.utils.mjwp import _object_mesh_body_id
from retargeting.utils.viser_viewer import _mujoco_mesh_to_trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/sharpa/left/cupmove/0")
    p.add_argument("--side", default="left")
    p.add_argument(
        "--heatmap",
        default="/home/intern/do-as-i-do/reconstruction/heatmap_out/contact_heatmap_palmar.npz",
    )
    p.add_argument("--clearance", type=float, default=0.2)
    p.add_argument("--conf-thresh", type=float, default=0.5)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _hand_body_ids(model: mujoco.MjModel, side: str, obj_body_id: int) -> list[int]:
    out = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name.startswith(f"{side}_") or bid == obj_body_id:
            continue
        if not body_has_mesh_geom(model, bid):
            continue
        out.append(bid)
    return out


def _world_mesh(model, data, body_id: int, rgba) -> list[trimesh.Trimesh]:
    out = []
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        m = _mujoco_mesh_to_trimesh(model, gi)
        if m is None:
            continue
        R = data.geom_xmat[gi].reshape(3, 3)
        t = data.geom_xpos[gi]
        m.apply_transform(np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]))
        m.visual = trimesh.visual.ColorVisuals(
            mesh=m, vertex_colors=np.tile(rgba, (len(m.vertices), 1))
        )
        out.append(m)
    return out


def _add_sphere(scene, center, rgba, radius=0.006):
    s = trimesh.creation.icosphere(radius=radius, subdivisions=2)
    s.apply_translation(center)
    s.visual = trimesh.visual.ColorVisuals(
        mesh=s, vertex_colors=np.tile(rgba, (len(s.vertices), 1))
    )
    scene.add_geometry(s)


def _add_arrow(scene, p0, p1, rgba, radius=0.003):
    vec = np.asarray(p1) - np.asarray(p0)
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, vec / length)
    angle = np.arccos(np.clip(np.dot(z, vec / length), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-8:
        R = trimesh.transformations.rotation_matrix(angle, axis)
    else:
        R = np.eye(4) if angle < 1e-6 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    cyl.apply_transform(R)
    cyl.apply_translation((np.asarray(p0) + np.asarray(p1)) / 2.0)
    cyl.visual = trimesh.visual.ColorVisuals(
        mesh=cyl, vertex_colors=np.tile(rgba, (len(cyl.vertices), 1))
    )
    scene.add_geometry(cyl)
    head = trimesh.creation.cone(radius=radius * 2.2, height=radius * 6, sections=16)
    head.apply_transform(R)
    head.apply_translation(np.asarray(p1))
    head.visual = trimesh.visual.ColorVisuals(
        mesh=head, vertex_colors=np.tile(rgba, (len(head.vertices), 1))
    )
    scene.add_geometry(head)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    side = args.side
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))

    kin = np.load(str(run_dir / "trajectory_kinematic.npz"))
    qpos_kin = kin["qpos"].astype(np.float64)  # (T, nq)
    qpos_ref_0 = qpos_kin[0].copy()

    fk_pre = mujoco.MjData(model)
    fk_pre.qpos[:] = qpos_ref_0
    mujoco.mj_kinematics(model, fk_pre)

    obj_body_id = _object_mesh_body_id(model, side)
    hand_bids = _hand_body_ids(model, side, obj_body_id)
    hand_pts = np.concatenate(
        [extract_body_mesh_verts(model, b, data=fk_pre, apply_geom_xform=True) for b in hand_bids],
        axis=0,
    )
    obj_verts_w = extract_body_mesh_verts(model, obj_body_id, data=fk_pre, apply_geom_xform=True)

    hand_centroid = hand_pts.mean(axis=0)
    obj_centroid = obj_verts_w.mean(axis=0)

    # --- direction 1: current centroid method ---
    away = hand_centroid - obj_centroid
    dir_centroid = away / np.linalg.norm(away)

    # --- direction 2: handle direction from heatmap high-confidence verts ---
    heat = np.load(args.heatmap)
    conf = heat["contact_confidence"]
    gi_vis = _find_geom_by_mesh_name(model, f"{side}_visual")
    assert gi_vis >= 0, f"{side}_visual mesh geom not found"
    mesh_id = model.geom_dataid[gi_vis]
    adr, n = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    vis_local = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
    assert vis_local.shape[0] == conf.shape[0], "heatmap/visual-mesh vertex count mismatch"
    R_vis = fk_pre.geom_xmat[gi_vis].reshape(3, 3)
    t_vis = fk_pre.geom_xpos[gi_vis]
    vis_world = vis_local @ R_vis.T + t_vis

    high = conf > args.conf_thresh
    handle_center = vis_world[high].mean(axis=0)
    dir_handle = handle_center - obj_centroid
    dir_handle = dir_handle / np.linalg.norm(dir_handle)

    ang = np.degrees(
        np.arccos(np.clip(np.dot(dir_centroid, dir_handle), -1.0, 1.0))
    )
    print(f"handle verts (conf>{args.conf_thresh}): {int(high.sum())}")
    print(f"dir_centroid (current) = {dir_centroid}")
    print(f"dir_handle   (proposed) = {dir_handle}")
    print(f"angle between the two backoff directions: {ang:.1f} deg")

    # --- pose the hand for both backoffs (wrist translate + floor clamp) ---
    def backoff_qpos(direction: np.ndarray) -> np.ndarray:
        q = qpos_ref_0.copy()
        q[0:3] = q[0:3] + args.clearance * direction
        q[2] = max(q[2], 0.01)
        return q

    fk_centroid = mujoco.MjData(model)
    fk_centroid.qpos[:] = backoff_qpos(dir_centroid)
    mujoco.mj_kinematics(model, fk_centroid)

    fk_handle = mujoco.MjData(model)
    fk_handle.qpos[:] = backoff_qpos(dir_handle)
    mujoco.mj_kinematics(model, fk_handle)

    # --- build scene ---
    scene = trimesh.Scene()

    # object colored by confidence: gray -> red
    lo = np.array([170, 170, 170], dtype=np.float64)
    hi = np.array([230, 40, 40], dtype=np.float64)
    cols = (lo[None, :] * (1 - conf[:, None]) + hi[None, :] * conf[:, None]).astype(np.uint8)
    rgba = np.concatenate([cols, np.full((len(cols), 1), 255, dtype=np.uint8)], axis=1)
    obj_mesh = trimesh.Trimesh(vertices=vis_world, faces=heat["faces"], process=False)
    obj_mesh.visual = trimesh.visual.ColorVisuals(mesh=obj_mesh, vertex_colors=rgba)
    scene.add_geometry(obj_mesh)

    ORANGE = np.array([255, 140, 40, 120], dtype=np.uint8)
    GREEN = np.array([60, 200, 90, 255], dtype=np.uint8)
    BLUE = np.array([70, 110, 240, 255], dtype=np.uint8)
    YELLOW = np.array([250, 200, 30, 255], dtype=np.uint8)
    MAGENTA = np.array([220, 50, 200, 255], dtype=np.uint8)

    for bid in hand_bids:
        for m in _world_mesh(model, fk_pre, bid, ORANGE):
            scene.add_geometry(m)
        for m in _world_mesh(model, fk_centroid, bid, GREEN):
            scene.add_geometry(m)
        for m in _world_mesh(model, fk_handle, bid, BLUE):
            scene.add_geometry(m)

    _add_arrow(scene, hand_centroid, hand_centroid + args.clearance * dir_centroid, YELLOW)
    _add_arrow(scene, hand_centroid, hand_centroid + args.clearance * dir_handle, MAGENTA)
    _add_sphere(scene, handle_center, MAGENTA, radius=0.008)
    _add_sphere(scene, obj_centroid, np.array([40, 90, 220, 255], dtype=np.uint8))

    # demo wrist path: light -> dark over time
    T = qpos_kin.shape[0]
    for t in range(T):
        f = t / max(T - 1, 1)
        c = np.array([250 - 180 * f, 250 - 180 * f, 250 - 100 * f, 255], dtype=np.uint8)
        _add_sphere(scene, qpos_kin[t, 0:3], c, radius=0.0035)

    out_path = Path(args.out) if args.out else run_dir / "warmup_fix_check" / "backoff_compare.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
