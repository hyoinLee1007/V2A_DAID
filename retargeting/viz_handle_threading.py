"""Does the robot's hand go through the mug handle the way the human's does?

The demonstration threads the index and middle fingers through the handle. That
is the thing the whole pipeline is trying to reproduce, and it is a yes/no
question that no scalar reward reports. This puts the three hands that matter
side by side, each with its own cup, at one frame:

  1  MANO, ray-corrected     what the reconstruction says the human did
  2  robot, IK reference     what retargeting turned that into, before physics
  3  robot, optimized        what the sampling optimizer actually produced

Reading it: if panel 2 already fails to thread the handle, no reward can fix it
— the reference the optimizer tracks is wrong, and the fix belongs in
retargeting. If panel 2 threads and panel 3 does not, the reference was fine
and the optimizer walked away from it.

Every panel is drawn in the *cup's* frame, so all three cups come out in
exactly the same pose and the only thing that differs between panels is the
hand. Drawing each in its own world frame instead leaves the reconstruction's
camera frame and the compiled scene's frame related by an unknown yaw, and the
cups then face different ways for no reason that has anything to do with the
grasp. The mug's own +y is its up axis, so its local frame needs no further
correction to display upright.

    python viz_handle_threading.py --frame 32
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
# hand_mesh lives with the correspondence viewers, which are grouped by topic
# under script/ rather than sitting next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent
                       / "script" / "MANO_Sharpa_heatmap_compare"))

# One colour per panel's hand, so MANO and the two robot poses stay separable
# when the panels sit side by side. The cup is deliberately the only warm-neutral
# thing on screen, since it is the one object that is identical in every panel.
CUP_RGBA = np.array([196, 176, 148, 255], np.uint8)          # sand
PANEL_HAND_RGBA = {
    "mano": np.array([235, 120,  70, 255], np.uint8),        # orange
    "reference": np.array([ 70, 140, 235, 255], np.uint8),   # blue
    "optimized": np.array([ 60, 190, 130, 255], np.uint8),   # green
}
HAND_RGBA = np.array([210, 210, 216, 255], np.uint8)         # fallback


def hand_colour(label: str) -> np.ndarray:
    low = label.lower()
    for key, rgba in PANEL_HAND_RGBA.items():
        if key in low:
            return rgba
    return HAND_RGBA


# Gap between panels as a multiple of each panel's own width. 1.25 left the
# hands nearly touching across the seam.
PANEL_GAP = 1.7


def to_cup_frame(pts, R_cup, t_cup):
    """World -> the cup's own frame, where its +y is up and its yaw is fixed."""
    return (pts - t_cup) @ R_cup


def recon_to_scene_rotation(recon_verts, scene_verts):
    """Rigid transform taking the reconstruction's cup mesh onto the compiled one.

    The scene's ``{side}_visual`` is the reconstruction's ``cup.obj`` with the
    gravity alignment baked into the vertices, so the same mesh sits in two
    different local frames — bbox [0.106, 0.113, 0.142] against
    [0.106, 0.143, 0.157]. Putting every panel in "the cup's frame" is only
    meaningful once that is undone.

    Vertex order is shared between the two files, so this is an exact fit
    rather than a registration; the caller checks the residual to be sure.
    """
    cx, cy = recon_verts.mean(0), scene_verts.mean(0)
    X, Y = recon_verts - cx, scene_verts - cy
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    # Translation too: the target is the BODY-frame mesh, which differs from the
    # raw one by geom_pos as well as geom_quat, so a rotation alone leaves an
    # 18.9 mm offset.
    t = cy - cx @ R.T
    return R, t, float(np.abs(scene_verts - (recon_verts @ R.T + t)).max())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/sharpa/left/cupmove/0")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--frame", type=int, default=32, help="Raw reconstruction frame.")
    p.add_argument("--raw-dir", default="../reconstruction/dataset/cupmove")
    p.add_argument("--task", default="cupmove")
    p.add_argument("--object", default="cup")
    p.add_argument("--zstar", default="../reconstruction/heatmap_out/oracle_zstar_cupmove.npz")
    p.add_argument("--mjwp", default=None,
                   help="Optimized trajectory npz (default: <run>/trajectory_mjwp.npz). "
                        "Pass 'none' to leave out panel 3.")
    p.add_argument("--out", default="outputs/handle_threading.glb")
    return p.parse_args()


def load_obj_text(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v, float), np.asarray(f, np.int64)


def coloured(v, f, rgba):
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m, vertex_colors=np.tile(rgba, (len(v), 1)))
    return m


def object_visual(model, data, side):
    """The object's visual mesh in world coords — not its 32 collision pieces."""
    import mujoco
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, f"{side}_visual")
    gi = next(g for g in range(model.ngeom) if int(model.geom_dataid[g]) == mid)
    b = int(model.geom_bodyid[gi])
    fa, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    # Body frame, i.e. with geom_pos/geom_quat applied. left_visual's geom is
    # rotated 132.1 deg from its body, so posing the raw mesh vertices drew the
    # cup 132 deg away from where the robot hand actually meets it — the hand
    # appeared to grip the base.
    v = _visual_mesh_local_verts(model, side)
    f = np.asarray(model.mesh_face[fa:fa + nf], np.int64)
    return v @ data.xmat[b].reshape(3, 3).T + data.xpos[b], f


def main() -> None:
    import mujoco
    from retargeting.utils.mano_robot_map import _bid
    from viz_mano_robot_map import hand_mesh

    args = parse_args()
    run = Path(args.run)
    side, f = args.side, args.frame
    raw = Path(args.raw_dir).resolve()

    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)

    # --- 1: MANO + cup, ray-corrected, in the camera frame ---
    meshes = np.load(f"{raw}/{args.task}/all_hand_meshes.npz")
    hv = meshes[f"{side}_vertices"].astype(float)[f]
    hf = meshes[f"{side}_faces"].astype(np.int64)
    z = np.load(args.zstar)
    ray = hv.mean(0)
    hv = hv + float(np.nan_to_num(z["z_star"][f])) * (ray / np.linalg.norm(ray))

    ov, of = load_obj_text(sorted(glob.glob(
        f"{raw}/video_segmentation/masks/frame_*_masks/{args.object}/"
        f"{args.object}.obj"))[0])
    layout = json.load(open(f"{raw}/obj_tracking_out/{args.object}/"
                            "combined_visualization/layout_camera_frame_optimized.json"))
    pose = next(o["local_to_scene"] for o in layout["objects"]
                if o.get("frame_idx", o.get("frame_index")) == f)
    R = Rotation.from_quat(np.asarray(pose["quat_wxyz_camera_frame"], float)[[1, 2, 3, 0]]).as_matrix()
    t_cup = np.asarray(pose["translation_camera_frame"], float)
    ov_m = ov * float(z["mesh_scale"])
    ow = ov_m @ R.T + t_cup

    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    scene_local = _visual_mesh_local_verts(model, side)
    nv = len(scene_local)
    if nv != len(ov_m):
        raise SystemExit(f"{side}_visual has {nv} verts, cup.obj has {len(ov_m)}")
    R_bake, t_bake, resid = recon_to_scene_rotation(ov_m, scene_local)
    print(f"reconstruction -> scene mesh rotation fitted, residual {resid:.2e} m")
    if resid > 1e-5:
        raise SystemExit("the two cup meshes are not the same mesh rotated; "
                         "panels would not be comparable")

    panels.append(("MANO (ray-corrected)",
                   to_cup_frame(hv, R, t_cup) @ R_bake.T + t_bake, hf,
                   to_cup_frame(ow, R, t_cup) @ R_bake.T + t_bake, of))

    # --- 2 and 3: the robot, in the compiled scene's frame ---

    def robot_panel(label, qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        rv, rf = hand_mesh(model, data, side)
        cv, cf = object_visual(model, data, side)
        ob = _bid(model, f"{side}_object")
        R_cup, t_cup = data.xmat[ob].reshape(3, 3), data.xpos[ob].copy()
        panels.append((label,
                       to_cup_frame(rv, R_cup, t_cup), rf,
                       to_cup_frame(cv, R_cup, t_cup), cf))

    kin = np.load(run / "trajectory_kinematic.npz")["qpos"]
    # kinematic[i] is raw frame i — verified by Procrustes-matching the object
    # translation sequences (offset 0 beat every other by >2x).
    if not 0 <= f < len(kin):
        raise SystemExit(f"frame {f} outside the IK reference ({len(kin)} frames)")
    robot_panel("robot: IK reference", kin[f])

    mjwp = args.mjwp if args.mjwp is not None else str(run / "trajectory_mjwp.npz")
    if mjwp.lower() != "none" and Path(mjwp).exists():
        t = np.load(mjwp, allow_pickle=True)
        q = np.asarray(t["qpos"], float).reshape(-1, model.nq)
        # Steps map to frames linearly after warmup. Matching by object pose
        # instead looks more principled and is not: this cup barely moves for
        # the first 32 frames (4.5 cm, against 15 cm over the next 18), so the
        # nearest-object step is degenerate exactly over the grasp — every
        # early frame matches the very first step.
        warm = np.asarray(t["warmup_progress"], float).reshape(-1)
        per_chunk = len(q) // len(warm)
        w_end = (int(np.argmax(warm >= 1.0)) * per_chunk
                 if (warm >= 1.0).any() else 0)
        step = min(int(w_end + f * (len(q) - w_end) / len(kin)), len(q) - 1)
        err = float(np.linalg.norm(q[step, -7:-4] - kin[f, -7:-4]))
        print(f"optimized step {step}/{len(q)} for frame {f} "
              f"(warmup ends at {w_end}; object {100 * err:.2f} cm from the reference)")
        robot_panel("robot: optimized", q[step])

    scene = trimesh.Scene()
    widths = [np.ptp(np.vstack([h, c]), 0)[0] for _, h, _, c, _ in panels]
    x = 0.0
    for i, (label, h, hf_, c, cf_) in enumerate(panels):
        both = np.vstack([h, c])
        shift = np.array([x - both[:, 0].min(), 0.0, 0.0]) - np.array(
            [0.0, both[:, 1].mean(), both[:, 2].mean()])
        scene.add_geometry(coloured(h + shift, hf_, hand_colour(label)))
        scene.add_geometry(coloured(c + shift, cf_, CUP_RGBA))
        print(f"panel {i + 1}: {label}   hand rgb {tuple(hand_colour(label)[:3])}")
        x += widths[i] * PANEL_GAP

    scene.export(args.out)
    print(f"\nSaved {args.out}   (frame {f}; left to right: "
          + ", ".join(p[0] for p in panels) + ")")


if __name__ == "__main__":
    main()
