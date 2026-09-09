"""Phase B: derive a contact heatmap from the rectified hand-object geometry.

This is CHOIR's "Initial anchor construction" (supplementary 7.2), run on top of
the Phase A oracle correction instead of a trained Stage-2 model. Per frame:

    hand_v <- hand_v + z*(frame) * camera_ray          # Phase A rectification
    for each hand vertex:
        find the nearest object surface sample
        accept iff  distance < 2 cm  AND  angle(hand normal, hand->object) < 60 deg
        record the accepted point as (face id, barycentric)

Accepted anchors are accumulated on the object mesh — each anchor adds its
barycentric weights to the three vertices of its face — and the total is
normalized to [0, 1]. The result is the same quantity the manual annotation
tool produces, but derived from the demonstration instead of drawn by hand.

Two properties matter downstream and are enforced here:

- **Vertex order.** Output vertices/faces are read from the source ``.obj``
  text in file order. The retargeting pipeline's ``visual.obj`` is that same
  file scaled by ``mesh_scale`` line-by-line, so the indices line up 1:1 with
  the compiled ``{side}_visual`` mesh with no remapping. (Loading the mesh
  through Open3D re-indexes vertices and silently breaks this — that is how the
  original manual heatmap ended up misaligned.)
- **Frame selection.** Only frames the oracle marked ``good`` are used, and
  optionally only those whose post-correction fingertip gap is small. Approach,
  release and tracking-dropout frames otherwise inject contacts that never
  happened.

The normal cone is the geometric form of "only the palm side may touch". Note
it acts here as a *filter on correspondences*, not as an optimization penalty —
which is why it cannot create the escape-to-no-contact failure mode that the
equivalent reward term did.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

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
    p.add_argument("--no-correction", action="store_true",
                   help="Skip the ray-depth correction (ablation: what the raw "
                        "reconstruction alone would yield).")
    p.add_argument("--max-gap-after", type=float, default=1.0,
                   help="Additionally drop frames whose post-correction fingertip "
                        "gap exceeds this (cm); catches frames the 1-D search "
                        "could not fix at all.")
    p.add_argument("--frames", default=None,
                   help="Explicit frame range 'start:end' overriding auto-selection.")
    p.add_argument("--surface-samples", type=int, default=60000)
    p.add_argument("--palmar-only", default=None, metavar="MAP_NPZ",
                   help="Keep only correspondences on the palm side of the hand, "
                        "using the per-vertex palmar coordinate in a map built by "
                        "retargeting/build_mano_robot_map.py. The distance and "
                        "normal-cone tests do not imply palm-side contact: a "
                        "finger threaded through a mug handle has the handle "
                        "wrapped all the way around it, so its back sits within "
                        "2 cm with its normal pointing at the object just as its "
                        "pad does, and both are accepted. The demonstration has "
                        "no back-of-hand contact, so those pairs are geometry, "
                        "not evidence.")
    p.add_argument("--palmar-thresh", type=float, default=0.0,
                   help="Palmar coordinate a vertex must exceed. 0 is the "
                        "midline of the segment; raise it to keep only the "
                        "pad proper.")
    p.add_argument("--sigma-smooth", type=float, default=0.0,
                   help="Optional geodesic-free Gaussian smoothing radius (m) over "
                        "the mesh; 0 disables. Softens the map the way the manual "
                        "tool's falloff does.")
    p.add_argument("--out", default="heatmap_out/contact_heatmap_choir.npz")
    return p.parse_args()


def load_obj_text(path: str):
    """Vertices and triangles in the file's own order (never via a mesh loader)."""
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def load_object_poses(raw_dir: str, obj_name: str, n_frames: int):
    path = (f"{raw_dir}/obj_tracking_out/{obj_name}/combined_visualization/"
            "layout_camera_frame_optimized.json")
    layout = json.load(open(path))
    trans = np.zeros((n_frames, 3)); quat = np.zeros((n_frames, 4)); quat[:, 0] = 1.0
    valid = np.zeros(n_frames, dtype=bool)
    for obj in layout["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if fi is None or not (0 <= fi < n_frames):
            continue
        ls = obj["local_to_scene"]
        t = np.asarray(ls["translation_camera_frame"], dtype=float)
        if t[2] <= 0:
            continue
        trans[fi] = t; quat[fi] = np.asarray(ls["quat_wxyz_camera_frame"], float)
        valid[fi] = True
    return trans, quat, valid


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def barycentric(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of points p within their triangles tri (n,3,3)."""
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    v2 = p - tri[:, 0]
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-20)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    return np.clip(np.stack([u, v, w], axis=1), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    raw = os.path.abspath(args.raw_dir)

    meshes = np.load(f"{raw}/{args.task}/all_hand_meshes.npz")
    hv_all = meshes[f"{args.side}_vertices"].astype(np.float64)
    hf = meshes[f"{args.side}_faces"].astype(np.int64)
    n_frames = hv_all.shape[0]

    z = np.load(args.zstar)
    zstar = z["z_star"]; good = z["good"]; scale = float(z["mesh_scale"])
    gap_after = z["gap_after"] if "gap_after" in z.files else np.zeros(n_frames)

    obj_files = sorted(glob.glob(
        f"{raw}/video_segmentation/masks/frame_*_masks/{args.object}/{args.object}.obj"))
    if not obj_files:
        raise FileNotFoundError("object .obj not found")
    ov_norm, of = load_obj_text(obj_files[0])       # file order preserved
    ov = ov_norm * scale                            # metric, matches visual.obj
    print(f"object mesh: {ov.shape[0]} verts / {of.shape[0]} faces, "
          f"scale={scale:.6f}, bbox={np.round(np.ptp(ov, 0), 4)} m")

    trans, quat, valid = load_object_poses(raw, args.object, n_frames)

    if args.frames:
        a, b = (int(x) for x in args.frames.split(":"))
        use = np.zeros(n_frames, bool); use[a:b] = True
        use &= valid
    else:
        use = good & valid & (np.nan_to_num(gap_after, nan=1e9) <= args.max_gap_after)
    idx = np.flatnonzero(use)
    print(f"frames used: {len(idx)} / {n_frames}   {idx.min() if len(idx) else '-'}"
          f"..{idx.max() if len(idx) else '-'}")
    if len(idx) == 0:
        raise SystemExit("no usable frames")

    palmar = None
    if args.palmar_only:
        pm = np.load(args.palmar_only, allow_pickle=True)
        palmar = np.asarray(pm["mano_palmar"], float)
        if len(palmar) != hv_all.shape[1]:
            raise SystemExit(f"palmar map has {len(palmar)} vertices, hand has "
                             f"{hv_all.shape[1]}")
        print(f"palm-side filter on: {(palmar > args.palmar_thresh).sum()}"
              f"/{len(palmar)} MANO vertices eligible")

    accum = np.zeros(ov.shape[0])
    # The hand half of each anchor. Accumulating it costs nothing and is what
    # carries finger identity: the object map alone cannot say whether the
    # handle was held by the pad of a finger or the back of it, because both
    # land on the same object vertices.
    hand_accum = np.zeros(hv_all.shape[1])
    # The anchor is a *pair* — MANO vertex i met object vertex v — and both
    # np.add.at calls below throw that away, each writing into its own 1-D
    # histogram. Nothing then records that the two entries were the same event,
    # so a reward can only ask "is this skin point near *any* touched vertex".
    # That lets the index and middle fingers collapse onto one spot on the
    # handle, which is exactly what they do. Keeping the pair costs one dict.
    pair_accum: dict[tuple[int, int], float] = {}
    n_anchor_total = 0
    n_dropped_dorsal = 0
    cos_thresh = np.cos(np.radians(ANCHOR_CONE_DEG))

    for f in idx:
        R = Rotation.from_quat(quat[f][[1, 2, 3, 0]]).as_matrix()
        ow = ov @ R.T + trans[f]
        m = trimesh.Trimesh(vertices=ow, faces=of, process=False)
        pts, fid = trimesh.sample.sample_surface(m, args.surface_samples)
        tree = cKDTree(pts)

        verts = hv_all[f]
        if not args.no_correction:
            ray = verts.mean(0); ray = ray / np.linalg.norm(ray)
            verts = verts + zstar[f] * ray
        hn = vertex_normals(verts, hf)

        d, j = tree.query(verts)
        near = d < ANCHOR_MAX_DIST
        if not near.any():
            continue
        tgt = pts[j[near]]
        direction = tgt - verts[near]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        ok = (direction * hn[near]).sum(1) > cos_thresh
        if palmar is not None:
            before = int(ok.sum())
            ok &= palmar[near] > args.palmar_thresh
            n_dropped_dorsal += before - int(ok.sum())
        if not ok.any():
            continue

        anchor_pts = tgt[ok]
        anchor_face = fid[j[near]][ok]
        bary = barycentric(anchor_pts, ow[of[anchor_face]])
        anchor_hand = np.flatnonzero(near)[ok]
        for k in range(3):
            np.add.at(accum, of[anchor_face][:, k], bary[:, k])
            # Same splat, but keyed by the pair rather than by the object
            # vertex alone, so "which hand vertex met this object vertex"
            # survives the loop.
            for h, v, b in zip(anchor_hand, of[anchor_face][:, k], bary[:, k]):
                pair_accum[(int(h), int(v))] = pair_accum.get((int(h), int(v)), 0.0) + float(b)
        np.add.at(hand_accum, anchor_hand, 1.0)
        n_anchor_total += int(ok.sum())

    if accum.max() <= 0:
        raise SystemExit("no anchors accepted — check frame selection / correction")
    H = accum / accum.max()

    if args.sigma_smooth > 0:
        tree_v = cKDTree(ov)
        pairs = tree_v.query_ball_point(ov, r=3 * args.sigma_smooth)
        Hs = np.zeros_like(H)
        for i, nb in enumerate(pairs):
            nb = np.asarray(nb)
            w = np.exp(-((np.linalg.norm(ov[nb] - ov[i], axis=1) ** 2)
                         / (2 * args.sigma_smooth ** 2)))
            Hs[i] = (H[nb] * w).sum() / max(w.sum(), 1e-12)
        H = Hs / max(Hs.max(), 1e-12)

    Hh = hand_accum / max(hand_accum.max(), 1e-12)
    # Sparse (hand vertex, object vertex, weight) triples: the anchors as pairs.
    # Note the weights are the *unsmoothed* splat counts — smoothing H spreads
    # confidence over neighbouring vertices for the object-side field, but a
    # target list wants the vertices actually touched, not their neighbours.
    if pair_accum:
        pk = np.array(sorted(pair_accum), dtype=np.int64).reshape(-1, 2)
        pw = np.array([pair_accum[(int(a), int(b))] for a, b in pk], dtype=np.float32)
    else:
        pk, pw = np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
    print(f"anchor pairs kept: {len(pk)} distinct (hand, object) vertex pairs "
          f"over {len(np.unique(pk[:, 0])) if len(pk) else 0} hand vertices")

    np.savez(args.out, vertices=ov, faces=of, contact_confidence=H.astype(np.float32),
             pair_hand=pk[:, 0], pair_object=pk[:, 1], pair_weight=pw,
             binary_contact=(H > 0.5).astype(np.uint8),
             hand_confidence=Hh.astype(np.float32), hand_counts=hand_accum,
             frames_used=idx, n_anchors=n_anchor_total, mesh_scale=scale,
             anchor_max_dist=ANCHOR_MAX_DIST, anchor_cone_deg=ANCHOR_CONE_DEG,
             palmar_only=bool(args.palmar_only), palmar_thresh=args.palmar_thresh)

    print(f"\nanchors accepted: {n_anchor_total}  ({n_anchor_total / len(idx):.0f} per frame)")
    if palmar is not None:
        print(f"  back-of-hand pairs dropped: {n_dropped_dorsal} "
              f"({100 * n_dropped_dorsal / max(n_dropped_dorsal + n_anchor_total, 1):.1f}%"
              " of what distance+cone alone accepted)")
    for t in (0.0, 0.25, 0.5, 0.9):
        print("  H > %.2f : %6d verts (%.1f%%)" % (t, (H > t).sum(), 100 * (H > t).mean()))
    hi = H > 0.5
    if hi.any():
        print("  H>0.5 region bbox extent = %s m" % np.round(np.ptp(ov[hi], 0), 4))
    print("\nhand side: %d of %d MANO vertices took part"
          % (int((hand_accum > 0).sum()), len(hand_accum)))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
