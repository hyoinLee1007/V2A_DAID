"""Show the anchor pairs themselves: which hand point met which object point.

Every other viewer in this repo shows one side or the other as a heat field —
"the cup was touched here", "the hand touched with these vertices". Neither
shows the *correspondence*, which is the thing the reward now depends on. This
draws it: a stick from each participating MANO vertex to the weighted centroid
of the object vertices it actually met, with both ends coloured by finger.

Read it as: same colour = same finger; a stick is one entry of the pairing that
``extract_contact_map.py`` saves as ``pair_hand``/``pair_object``. Fingers whose
sticks fan out over a wide patch slid during the grasp; fingers whose sticks
converge held one spot.

The pose shown is the ray-corrected reconstruction at ``--frame``, i.e. exactly
the geometry the pairs were extracted from, so a stick that looks wrong is a
pairing that *is* wrong.

    python viz_anchor_pairs.py --frame 50
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# One colour per finger, shared by both ends of every stick.
FINGER_RGB = {
    "thumb":  (232,  92,  60),
    "index":  ( 54, 126, 232),
    "middle": ( 46, 176, 106),
    "ring":   (196, 110, 220),
    "pinky":  (232, 176,  50),
    "palm":   (130, 138, 150),
}
HAND_RGBA = np.array([224, 224, 230, 255], np.uint8)
CUP_RGBA = np.array([168, 170, 176, 255], np.uint8)

# OpenCV camera convention (+y down, +z forward) -> glTF viewer (+y up).
_VIEW_FIX = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="dataset/cupmove")
    p.add_argument("--task", default="cupmove")
    p.add_argument("--object", default="cup")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--frame", type=int, default=50)
    p.add_argument("--contact", default="heatmap_out/contact_heatmap_palmar.npz")
    p.add_argument("--zstar", default="heatmap_out/oracle_zstar_cupmove.npz")
    p.add_argument("--map", default="../retargeting/outputs/mano_robot_map_left.npz",
                   help="mano_robot_map npz, for the per-vertex finger labels.")
    p.add_argument("--min-weight", type=float, default=0.0,
                   help="Drop pairs below this share of the heaviest pair.")
    p.add_argument("--all-targets", action="store_true",
                   help="One stick per pair instead of one per hand vertex. "
                        "Shows the full fan-out; much busier.")
    p.add_argument("--stick-radius", type=float, default=0.0007)
    p.add_argument("--dot-radius", type=float, default=0.0022)
    p.add_argument("--out", default="heatmap_out/anchor_pairs.glb")
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
        if obj.get("frame_idx", obj.get("frame_index")) != frame:
            continue
        ls = obj["local_to_scene"]
        t = np.asarray(ls["translation_camera_frame"], float)
        q = np.asarray(ls["quat_wxyz_camera_frame"], float)
        if t[2] > 0:
            return t, q
    raise SystemExit(f"no object pose for frame {frame}")


def coloured(v, f, rgba):
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m, vertex_colors=np.tile(rgba, (len(v), 1)).astype(np.uint8))
    return m


def spheres(centers, colors, radius):
    base = trimesh.creation.icosphere(radius=radius, subdivisions=1)
    nb = len(base.vertices)
    v = np.tile(base.vertices, (len(centers), 1)) + np.repeat(centers, nb, 0)
    f = np.concatenate([base.faces + i * nb for i in range(len(centers))])
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.repeat(colors, nb, 0))
    return m


def sticks(a, b, colors, radius):
    """One thin cylinder per (a[i], b[i]) pair, built by hand.

    trimesh's ``creation.cylinder(segment=...)`` builds one mesh per call and
    is far too slow for a few thousand of them; this instances a single unit
    cylinder and transforms the vertices directly.
    """
    unit = trimesh.creation.cylinder(radius=radius, height=1.0, sections=6)
    uv, uf = np.asarray(unit.vertices), np.asarray(unit.faces)
    nb = len(uv)
    d = b - a
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-6
    a, b, d, L, colors = a[ok], b[ok], d[ok], L[ok], colors[ok]
    z = d / L[:, None]
    # Any vector not parallel to z gives a usable first basis vector.
    tmp = np.tile([0.0, 0.0, 1.0], (len(z), 1))
    flip = np.abs((z * tmp).sum(1)) > 0.9
    tmp[flip] = [1.0, 0.0, 0.0]
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)                       # (n, 3, 3)

    scaled = np.tile(uv, (len(z), 1, 1))
    scaled[:, :, 2] *= L[:, None]                          # unit height -> |d|
    verts = np.einsum("nij,nvj->nvi", R, scaled) + ((a + b) / 2)[:, None, :]
    faces = np.concatenate([uf + i * nb for i in range(len(z))])
    m = trimesh.Trimesh(verts.reshape(-1, 3), faces, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.repeat(colors, nb, 0))
    return m


def main() -> None:
    args = parse_args()
    raw = Path(args.raw_dir).resolve()
    f = args.frame

    c = np.load(args.contact, allow_pickle=True)
    if "pair_hand" not in c.files:
        raise SystemExit(f"{args.contact} has no pair_hand; re-run "
                         "extract_contact_map.py (it now saves the pairs).")
    ph = np.asarray(c["pair_hand"], np.int64)
    po = np.asarray(c["pair_object"], np.int64)
    pw = np.asarray(c["pair_weight"], float)
    if args.min_weight > 0:
        m = pw >= args.min_weight * pw.max()
        ph, po, pw = ph[m], po[m], pw[m]
    scale = float(c["mesh_scale"])

    finger = np.load(args.map, allow_pickle=True)["finger"].astype(str)

    meshes = np.load(f"{raw}/{args.task}/all_hand_meshes.npz")
    hv = meshes[f"{args.side}_vertices"].astype(float)[f]
    hf = meshes[f"{args.side}_faces"].astype(np.int64)
    z = np.load(args.zstar)
    ray = hv.mean(0)
    hv = hv + float(np.nan_to_num(z["z_star"][f])) * (ray / np.linalg.norm(ray))

    ov, of = load_obj_text(sorted(glob.glob(
        f"{raw}/video_segmentation/masks/frame_*_masks/{args.object}/"
        f"{args.object}.obj"))[0])
    t, q = load_object_pose(str(raw), args.object, f)
    ow = (ov * scale) @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T + t

    # Endpoints. One stick per hand vertex by default: the object end is the
    # weight-weighted centroid of everything that vertex met, which is what the
    # reward's target list is centred on.
    if args.all_targets:
        A, B, W, HV = hv[ph], ow[po], pw, ph
    else:
        uniq = np.unique(ph)
        A = hv[uniq]
        B = np.stack([(ow[po[ph == h]] * pw[ph == h, None]).sum(0)
                      / pw[ph == h].sum() for h in uniq])
        W = np.array([pw[ph == h].sum() for h in uniq])
        HV = uniq

    rgb = np.array([FINGER_RGB.get(finger[h], (140, 140, 140)) for h in HV], float)
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255.0)], 1).astype(np.uint8)

    centre = np.vstack([ow, hv]).mean(0)
    view = lambda p: (p - centre) @ _VIEW_FIX.T                    # noqa: E731

    scene = trimesh.Scene()
    scene.add_geometry(coloured(view(hv), hf, HAND_RGBA))
    scene.add_geometry(coloured(view(ow), of, CUP_RGBA))
    scene.add_geometry(sticks(view(A), view(B), rgba, args.stick_radius))
    scene.add_geometry(spheres(view(A), rgba, args.dot_radius))
    scene.add_geometry(spheres(view(B), rgba, args.dot_radius))
    scene.export(args.out)

    print(f"frame {f}:  {len(ph)} pairs over {len(np.unique(ph))} hand vertices "
          f"and {len(np.unique(po))} object vertices")
    print(f"drawn: {len(A)} sticks"
          + ("  (one per pair)" if args.all_targets else "  (one per hand vertex)"))
    print(f"\n{'finger':8s} {'verts':>6s} {'pairs':>6s} {'stick len':>10s}")
    L = np.linalg.norm(B - A, axis=1)
    for name in ("thumb", "index", "middle", "ring", "pinky", "palm"):
        k = np.array([finger[h] == name for h in HV])
        if not k.any():
            continue
        npair = int(sum((finger[h] == name) for h in ph))
        print(f"{name:8s} {int(k.sum()):6d} {npair:6d} {L[k].mean() * 100:8.2f} cm")
    print(f"\nSaved {args.out}")
    print("colour = finger; a stick joins a MANO vertex to the object patch it met")


if __name__ == "__main__":
    main()
