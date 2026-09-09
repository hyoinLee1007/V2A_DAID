"""Show both halves of the contact map together: the object's and the hand's.

Every earlier visualisation showed one half. Seen alone neither says much — the
object map marks the handle without saying what held it, and the hand map marks
fingers without saying what they held. Side by side, in the pose the frame
actually had, they are one statement.

  left    cup + MANO hand at the chosen frame, after the ray-depth correction.
          The cup is shaded by ``contact_confidence`` and the hand by
          ``hand_confidence``, both grey -> saturated.
  right   the robot hand at its open pose, with the transferred contact points
          as spheres on the same colour scale.

Red is the object side and blue the hand side throughout, matching
``viz_oracle_3d.py``.

    python viz_contact_pair.py --frame 32
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

GREY = np.array([175.0, 175.0, 175.0])
RED = np.array([230.0, 45.0, 45.0])
BLUE = np.array([55.0, 95.0, 235.0])

# OpenCV camera convention (+y down, +z forward) -> glTF viewer (+y up).
_VIEW_FIX = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="dataset/cupmove")
    p.add_argument("--task", default="cupmove")
    p.add_argument("--object", default="cup")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--frame", type=int, default=32)
    p.add_argument("--contact", default="heatmap_out/contact_heatmap_palmar.npz")
    p.add_argument("--zstar", default="heatmap_out/oracle_zstar_cupmove.npz")
    p.add_argument("--robot-contact",
                   default="../retargeting/outputs/robot_contact_map_left.npz")
    p.add_argument("--no-correction", action="store_true")
    p.add_argument("--sphere-radius", type=float, default=0.0026)
    p.add_argument("--out", default="heatmap_out/contact_pair.glb")
    return p.parse_args()


def load_obj_text(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v, float), np.asarray(f, np.int64)


def load_object_pose(raw_dir, obj_name, frame):
    path = (f"{raw_dir}/obj_tracking_out/{obj_name}/combined_visualization/"
            "layout_camera_frame_optimized.json")
    for obj in json.load(open(path))["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if fi != frame:
            continue
        ls = obj["local_to_scene"]
        t = np.asarray(ls["translation_camera_frame"], float)
        q = np.asarray(ls["quat_wxyz_camera_frame"], float)
        if t[2] > 0:
            return t, q
    raise SystemExit(f"no object pose for frame {frame}")


def shade(mesh, conf, hot):
    """Grey -> ``hot`` with confidence, as vertex colours."""
    t = np.clip(conf, 0.0, 1.0)[:, None]
    rgb = GREY * (1 - t) + hot * t
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.concatenate(
            [rgb, np.full((len(rgb), 1), 255.0)], 1).astype(np.uint8))
    return mesh


def spheres(centers, colors, radius):
    base = trimesh.creation.icosphere(radius=radius, subdivisions=1)
    nb = len(base.vertices)
    v = np.tile(base.vertices, (len(centers), 1)) + np.repeat(centers, nb, 0)
    f = np.concatenate([base.faces + i * nb for i in range(len(centers))])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.repeat(colors, nb, 0))
    return m


def main() -> None:
    args = parse_args()
    raw = Path(args.raw_dir).resolve()
    f = args.frame

    c = np.load(args.contact)
    H_obj = np.clip(np.asarray(c["contact_confidence"], float), 0, 1)
    if "hand_confidence" not in c.files:
        raise SystemExit(f"{args.contact} has no hand_confidence; re-run "
                         "extract_contact_map.py")
    H_hand = np.clip(np.asarray(c["hand_confidence"], float), 0, 1)
    scale = float(c["mesh_scale"])

    meshes = np.load(f"{raw}/{args.task}/all_hand_meshes.npz")
    hv = meshes[f"{args.side}_vertices"].astype(float)[f]
    hf = meshes[f"{args.side}_faces"].astype(np.int64)

    if not args.no_correction:
        z = np.load(args.zstar)
        ray = hv.mean(0)
        hv = hv + float(np.nan_to_num(z["z_star"][f])) * (ray / np.linalg.norm(ray))

    ov, of = load_obj_text(sorted(glob.glob(
        f"{raw}/video_segmentation/masks/frame_*_masks/{args.object}/"
        f"{args.object}.obj"))[0])
    t, q = load_object_pose(str(raw), args.object, f)
    ow = (ov * scale) @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T + t

    scene = trimesh.Scene()
    view = lambda p: p @ _VIEW_FIX.T                              # noqa: E731

    pair_pts = np.vstack([ow, hv])
    centre = pair_pts.mean(0)
    scene.add_geometry(shade(
        trimesh.Trimesh(view(ow - centre), of, process=False), H_obj, RED))
    scene.add_geometry(shade(
        trimesh.Trimesh(view(hv - centre), hf, process=False), H_hand, BLUE))

    print(f"frame {f}:  object H>0.5 on {int((H_obj > 0.5).sum())} verts,  "
          f"hand contact on {int((H_hand > 0).sum())} of {len(H_hand)} verts")

    # --- right panel: the same hand map after the transfer to the robot ---
    rc = Path(args.robot_contact)
    if rc.exists():
        rt = Path(__file__).resolve().parent.parent / "retargeting"
        sys.path.insert(0, str(rt))
        # hand_mesh lives with the correspondence viewers, grouped by topic
        # under script/ rather than at the retargeting root.
        sys.path.insert(0, str(rt / "script" / "MANO_Sharpa_heatmap_compare"))
        import mujoco
        from retargeting.utils.mano_robot_map import _bid, _canonical, _palmar_world
        from viz_mano_robot_map import hand_mesh

        d = np.load(rc, allow_pickle=True)
        side = str(d["side"])
        # The map records the scene path as the builder was invoked with it,
        # i.e. relative to the retargeting directory, not to this one.
        scene_path = Path(str(d["scene"]))
        if not scene_path.is_absolute():
            scene_path = rt / scene_path
        rm = mujoco.MjModel.from_xml_path(str(scene_path))
        rd = _canonical(rm, side)
        body = d["body_id"].astype(np.int64)
        P = rd.xpos[body] + np.einsum(
            "nij,nj->ni", rd.xmat[body].reshape(-1, 3, 3),
            np.asarray(d["local_offset"], float))
        w = np.clip(np.asarray(d["weight"], float), 0, 1)[:, None]

        rv, rf = hand_mesh(rm, rd, side)
        # Same display orientation as viz_mano_robot_map: fingers up, palm out.
        o = rd.xpos[_bid(rm, f"{side}_hand_C_MC")]
        fwd = rd.xpos[_bid(rm, f"{side}_middle_MCP_VL")] - o
        fwd /= np.linalg.norm(fwd)
        pal = _palmar_world(rm, side, "middle")
        pal = pal - fwd * float(pal @ fwd)
        pal /= np.linalg.norm(pal)
        R = np.stack([np.cross(fwd, pal), fwd, pal], axis=0)

        gap = (np.ptp(view(pair_pts - centre), 0)[0]
               + np.ptp((rv - o) @ R.T, 0)[0]) * 0.75
        place = lambda p: view((p - o) @ R.T) + np.array([gap, 0.0, 0.0])  # noqa: E731

        h = trimesh.Trimesh(place(rv), rf, process=False)
        h.visual = trimesh.visual.ColorVisuals(
            mesh=h, vertex_colors=np.tile([200, 200, 200, 255], (len(rv), 1)).astype(np.uint8))
        scene.add_geometry(h)
        rgb = GREY * (1 - w) + BLUE * w
        scene.add_geometry(spheres(
            place(P), np.concatenate([rgb, np.full((len(rgb), 1), 255.0)], 1
                                     ).astype(np.uint8), args.sphere_radius))
        print(f"robot panel: {len(P)} transferred contact points")

    scene.export(args.out)
    print(f"\nSaved {args.out}")
    print("left  = cup (RED = touched) + MANO hand (BLUE = touching), frame "
          f"{f}, ray-corrected")
    print("right = robot hand, open pose, with the same contacts transferred")


if __name__ == "__main__":
    main()
