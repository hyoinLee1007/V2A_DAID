"""3D view of the oracle ray-depth correction and the contact anchors it yields.

Exports a GLB you can orbit in any glTF viewer. Per requested frame:

  left  panel : cup + hand BEFORE correction (orange)
  right panel : cup + hand AFTER  correction (green), shifted by z* along the ray
  red spheres : object-side anchors      blue spheres : the MANO vertices they pair with

Geometry is rotated 180 deg about x on export. The reconstruction lives in the
OpenCV camera convention (+y down, +z into the scene) while glTF viewers assume
+y up, so without this the cup shows up upside down and orbiting fights you.

Anchors use CHOIR's acceptance rule (within 2 cm, hand normal inside a 60 deg
cone of the hand->object direction) — the same rule extract_contact_map.py
accumulates into the heatmap, so this shows exactly which correspondences feed
that map.

Several frames can be shown at once; they are laid out along +x so the change
over time is visible in one scene.

    python viz_oracle_3d.py --frames 32
    python viz_oracle_3d.py --frames 32 --no-anchors     # placement only
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ANCHOR_MAX_DIST = 0.02
ANCHOR_CONE_DEG = 60.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="dataset/cupmove")
    p.add_argument("--task", default="cupmove")
    p.add_argument("--object", default="cup")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--zstar", default="heatmap_out/oracle_zstar_cupmove.npz")
    p.add_argument("--frames", default="32",
                   help="Comma-separated frame indices. Each frame is drawn as a "
                        "before|after pair.")
    p.add_argument("--no-anchors", action="store_true")
    p.add_argument("--palmar-only", default=None, metavar="MAP_NPZ",
                   help="Draw only palm-side anchors, using the per-vertex "
                        "palmar coordinate from a map built by "
                        "retargeting/build_mano_robot_map.py. Without it the "
                        "anchors shown are what distance and the normal cone "
                        "alone accept, which is roughly twice what the reward "
                        "actually receives — a finger threaded through the "
                        "handle passes both tests with its back as readily as "
                        "with its pad.")
    p.add_argument("--surface-samples", type=int, default=60000)
    p.add_argument("--sphere-radius", type=float, default=0.0022)
    p.add_argument("--out", default="heatmap_out/oracle_correction_3d.glb")
    return p.parse_args()


def load_obj_text(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v, float), np.asarray(f, np.int64)


def load_object_poses(raw_dir, obj_name, n):
    path = (f"{raw_dir}/obj_tracking_out/{obj_name}/combined_visualization/"
            "layout_camera_frame_optimized.json")
    layout = json.load(open(path))
    t = np.zeros((n, 3)); q = np.zeros((n, 4)); q[:, 0] = 1.0
    for obj in layout["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if fi is None or not (0 <= fi < n):
            continue
        ls = obj["local_to_scene"]
        tt = np.asarray(ls["translation_camera_frame"], float)
        if tt[2] <= 0:
            continue
        t[fi] = tt; q[fi] = np.asarray(ls["quat_wxyz_camera_frame"], float)
    return t, q


def vertex_normals(v, f):
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


# OpenCV camera convention (+y down, +z forward) -> glTF viewer (+y up).
_VIEW_FIX = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def to_view(p: np.ndarray) -> np.ndarray:
    return p @ _VIEW_FIX.T


def colored(mesh: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))
    return mesh


def spheres(centers, rgba, radius):
    if len(centers) == 0:
        return None
    base = trimesh.creation.icosphere(radius=radius, subdivisions=1)
    v = np.tile(base.vertices, (len(centers), 1)) + np.repeat(centers, len(base.vertices), 0)
    f = np.concatenate([base.faces + i * len(base.vertices) for i in range(len(centers))])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return colored(m, rgba)


def main() -> None:
    args = parse_args()
    raw = os.path.abspath(args.raw_dir)
    frames = [int(x) for x in args.frames.split(",")]

    meshes = np.load(f"{raw}/{args.task}/all_hand_meshes.npz")
    hv = meshes[f"{args.side}_vertices"].astype(np.float64)
    hf = meshes[f"{args.side}_faces"].astype(np.int64)
    n = hv.shape[0]

    z = np.load(args.zstar)
    scale = float(z["mesh_scale"])
    # Read the offsets the *pipeline* applies, not the raw solutions. The npz
    # holds a z* for every frame the scan touched, including ones the gate then
    # rejected; drawing those shows a correction the optimizer never sees.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "retargeting"))
    from retargeting.utils.ray_depth_correction import load_ray_depth
    zstar, _solved = load_ray_depth(args.zstar, n)
    ov_norm, of = load_obj_text(sorted(glob.glob(
        f"{raw}/video_segmentation/masks/frame_*_masks/{args.object}/{args.object}.obj"))[0])
    ov = ov_norm * scale
    trans, quat = load_object_poses(raw, args.object, n)

    GREY   = np.array([170, 170, 170, 255], np.uint8)
    ORANGE = np.array([255, 150, 50, 110], np.uint8)   # before, translucent
    GREEN  = np.array([60, 200, 90, 255], np.uint8)    # after
    RED    = np.array([230, 40, 40, 255], np.uint8)    # object-side anchors
    BLUE   = np.array([50, 90, 240, 255], np.uint8)    # hand-side anchors

    palmar = None
    if args.palmar_only:
        pm = np.load(args.palmar_only, allow_pickle=True)
        palmar = np.asarray(pm["mano_palmar"], float)
        if len(palmar) != hv.shape[1]:
            raise SystemExit(f"palmar map has {len(palmar)} vertices, hand has "
                             f"{hv.shape[1]}")
        print("palm-side filter on: %d/%d MANO vertices eligible"
              % (int((palmar > 0).sum()), len(palmar)))

    scene = trimesh.Scene()
    cos_t = np.cos(np.radians(ANCHOR_CONE_DEG))

    # Panel spacing must clear the *hand* too, not just the cup: sized from the
    # cup only, wide hand poses ran into the neighbouring panel.
    def panel_width(f):
        R = Rotation.from_quat(quat[f][[1, 2, 3, 0]]).as_matrix()
        allpts = np.vstack([ov @ R.T + trans[f], hv[f]])
        return float(np.ptp(allpts, 0)[0])

    width = max(panel_width(f) for f in frames if 0 <= f < n) * 1.35
    panel = 0

    for f in frames:
        if not (0 <= f < n):
            print(f"frame {f} out of range, skipped"); continue

        R = Rotation.from_quat(quat[f][[1, 2, 3, 0]]).as_matrix()
        ow = ov @ R.T + trans[f]
        cup = trimesh.Trimesh(vertices=ow, faces=of, process=False)
        tree_cup = cKDTree(ow)

        before = hv[f]
        ray = before.mean(0); ray = ray / np.linalg.norm(ray)
        dz = 0.0 if np.isnan(zstar[f]) else float(zstar[f])
        after = before + dz * ray

        pts = tree = None
        if not args.no_anchors:
            pts, _ = trimesh.sample.sample_surface(cup, args.surface_samples)
            tree = cKDTree(pts)

        def anchors_for(hand):
            """Accepted (object point, hand vertex) pairs under CHOIR's rule."""
            if tree is None:
                return np.zeros((0, 3)), np.zeros((0, 3))
            hn = vertex_normals(hand, hf)
            d, j = tree.query(hand)
            near = d < ANCHOR_MAX_DIST
            if not near.any():
                return np.zeros((0, 3)), np.zeros((0, 3))
            tgt = pts[j[near]]
            dirv = tgt - hand[near]
            dirv /= np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-12)
            ok = (dirv * hn[near]).sum(1) > cos_t
            if palmar is not None:
                ok &= palmar[near] > 0.0
            return tgt[ok], hand[near][ok]

        counts = []
        # Two panels per frame: BEFORE on the left, AFTER on its right.
        for hand, hand_col in ((before, ORANGE), (after, GREEN)):
            shift = np.array([panel * width, 0.0, 0.0])
            scene.add_geometry(colored(
                trimesh.Trimesh(to_view(ow) + shift, of, process=False), GREY))
            scene.add_geometry(colored(
                trimesh.Trimesh(to_view(hand) + shift, hf, process=False), hand_col))
            o_pts, h_pts = anchors_for(hand)
            counts.append(len(o_pts))
            for p_, col in ((o_pts, RED), (h_pts, BLUE)):
                s = spheres(to_view(p_) + shift, col, args.sphere_radius)
                if s is not None:
                    scene.add_geometry(s)
            panel += 1

        gap_b = tree_cup.query(before)[0].min() * 100
        gap_a = tree_cup.query(after)[0].min() * 100
        print("frame %3d | z* = %+6.2f cm | min hand-cup gap %5.2f -> %5.2f cm | "
              "anchors %d -> %d" % (f, dz * 100, gap_b, gap_a, counts[0], counts[1]))

    scene.export(args.out)
    print(f"\nSaved {args.out}")
    print("layout: per frame, LEFT = before / RIGHT = after")
    print("legend: grey cup | ORANGE hand = before | GREEN hand = after | "
          "RED = object-side anchors | BLUE = hand-side anchors")


if __name__ == "__main__":
    main()
