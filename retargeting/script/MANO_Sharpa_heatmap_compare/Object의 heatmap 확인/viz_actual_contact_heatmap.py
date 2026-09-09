"""Actual-contact heatmap from an executed trajectory, vs the defined one.

Replays trajectory_mjwp.npz kinematically (FK only, no physics) and marks, for
every object visual-mesh vertex, how often any hand-mesh point came within
``--thresh`` meters of it. Exports a single GLB with three copies of the
object side by side:

  left   : the DEFINED contact heatmap (contact_confidence, gray -> red)
  middle : the ACTUAL contact frequency from the trajectory (gray -> red)
  right  : overlap view — green = defined & touched, red = defined but never
           touched, blue = touched outside the defined region, gray = neither

Also prints precision/recall-style agreement metrics between the actual
contact set and the defined high-confidence region.

Run from retargeting/ in the `retargeting` conda env (CPU only):
    python viz_actual_contact_heatmap.py \
        --run-dir outputs/sharpa/left/cupmove_initial_20cm_backoffset/0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from retargeting.utils.contact_region_reward import _find_geom_by_mesh_name
from retargeting.utils.in_hand import body_has_mesh_geom, extract_body_mesh_verts
from retargeting.utils.mjwp import _object_mesh_body_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/sharpa/left/cupmove_initial_20cm_backoffset/0")
    p.add_argument("--side", default="left")
    p.add_argument(
        "--heatmap",
        default="/home/intern/do-as-i-do/reconstruction/heatmap_out/contact_heatmap_objorder.npz",
    )
    p.add_argument("--thresh", type=float, default=0.01, help="Contact distance threshold (m).")
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--conf-thresh", type=float, default=0.5, help="Defined-region cutoff.")
    p.add_argument("--freq-thresh", type=float, default=0.05,
                   help="Actual-contact cutoff as a fraction of the max per-vertex count.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def object_local_visual_verts(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, int]:
    gi = _find_geom_by_mesh_name(model, f"{side}_visual")
    assert gi >= 0, f"{side}_visual mesh geom not found"
    mesh_id = int(model.geom_dataid[gi])
    adr, n = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    v = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
    Rg = np.asarray(
        Rotation.from_quat(np.asarray(model.geom_quat[gi])[[1, 2, 3, 0]]).as_matrix()
    )
    return v @ Rg.T + np.asarray(model.geom_pos[gi]), gi


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    side = args.side
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    data = mujoco.MjData(model)

    traj = np.load(str(run_dir / "trajectory_mjwp.npz"))
    qpos = traj["qpos"].reshape(-1, traj["qpos"].shape[-1]).astype(np.float64)
    T = qpos.shape[0]

    obj_body_id = _object_mesh_body_id(model, side)
    hand_bids = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name.startswith(f"{side}_") or bid == obj_body_id:
            continue
        if body_has_mesh_geom(model, bid):
            hand_bids.append(bid)

    obj_local, _ = object_local_visual_verts(model, side)  # (V,3) object-body frame
    V = obj_local.shape[0]
    heat = np.load(args.heatmap)
    conf = np.asarray(heat["contact_confidence"], dtype=np.float64)
    assert conf.shape[0] == V, "heatmap not in visual-mesh order — run fix_heatmap_vertex_order.py"

    count = np.zeros(V, dtype=np.int64)
    frames = range(0, T, args.frame_stride)
    n_frames = 0
    for t in frames:
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        hand_pts = np.concatenate(
            [extract_body_mesh_verts(model, b, data=data, apply_geom_xform=True) for b in hand_bids],
            axis=0,
        )
        if hand_pts.shape[0] > 2000:
            hand_pts = hand_pts[:: (hand_pts.shape[0] + 1999) // 2000]
        # world -> object body frame
        pos = qpos[t, -7:-4]
        quat_wxyz = qpos[t, -4:]
        R = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        hand_local = (hand_pts - pos) @ R
        dd, _ = cKDTree(hand_local).query(obj_local, distance_upper_bound=args.thresh)
        count += np.isfinite(dd) & (dd < args.thresh)
        n_frames += 1

    if count.max() == 0:
        print("WARNING: no contact detected anywhere on the trajectory at "
              f"thresh={args.thresh}m — nothing to visualize.")
    freq = count / max(count.max(), 1)

    defined = conf > args.conf_thresh
    touched = freq > args.freq_thresh
    both = defined & touched
    print(f"frames sampled: {n_frames} (stride {args.frame_stride}), thresh={args.thresh}m")
    print(f"defined region (conf>{args.conf_thresh}): {defined.sum()} verts")
    print(f"actually touched (freq>{args.freq_thresh}): {touched.sum()} verts")
    print(f"overlap: {both.sum()} verts")
    if touched.sum():
        print(f"precision (touched verts inside defined region): {both.sum()/touched.sum():.3f}")
    if defined.sum():
        print(f"recall    (defined region actually touched):     {both.sum()/defined.sum():.3f}")
    # confidence-weighted: how much of the touching happened on high-conf area
    if count.sum():
        print(f"contact-weighted mean confidence: {(conf*count).sum()/count.sum():.3f} "
              f"(1.0 = all contact on handle center, ~0.1 = uniform)")

    # ---- GLB: three cups side by side ----
    faces = np.asarray(heat["faces"], dtype=np.int64)
    span = float(np.ptp(obj_local[:, 0])) * 1.6

    def ramp(vals: np.ndarray) -> np.ndarray:
        lo = np.array([170, 170, 170], dtype=np.float64)
        hi = np.array([230, 40, 40], dtype=np.float64)
        c = lo[None, :] * (1 - vals[:, None]) + hi[None, :] * vals[:, None]
        return np.concatenate([c, np.full((len(c), 1), 255.0)], axis=1).astype(np.uint8)

    overlap_rgba = np.tile(np.array([185, 185, 185, 255], dtype=np.uint8), (V, 1))
    overlap_rgba[defined & ~touched] = [220, 50, 50, 255]    # defined, never touched
    overlap_rgba[~defined & touched] = [60, 90, 230, 255]    # touched outside defined
    overlap_rgba[both] = [40, 190, 80, 255]                  # agreement

    scene = trimesh.Scene()
    for i, rgba in enumerate((ramp(conf), ramp(freq), overlap_rgba)):
        m = trimesh.Trimesh(vertices=obj_local + [i * span, 0, 0], faces=faces, process=False)
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
        scene.add_geometry(m)

    out_path = Path(args.out) if args.out else run_dir / "warmup_fix_check" / "actual_vs_defined_contact.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))
    print(f"Saved {out_path}")
    print("layout: left = defined heatmap, middle = actual contact, "
          "right = overlap (green=agree, red=defined-only, blue=touched-only)")


if __name__ == "__main__":
    main()
