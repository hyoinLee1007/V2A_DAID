"""Phase A: oracle ray-depth correction — the ceiling CHOIR Stage 2 could reach.

CHOIR's Stage 2 *predicts* a scalar offset along the camera ray that moves a
monocular hand estimate into a contact-plausible placement. Before training any
such model, this computes the same scalar **directly**, by 1-D search against
the object surface, and reports whether it is enough to fix the reconstruction.

If the oracle cannot fix a sequence, no model trained to approximate it will,
so this is a zero-training go/no-go gate. If it can, the corrected hands feed
the downstream anchor/contact-map work (Phase B) immediately, still with no
model.

Everything runs in the **camera frame**, where the ray is meaningful: the
camera sits at the origin looking down +z (process_dataset.py rejects object
poses with ``t[2] <= 0`` as "negative depth"), so the viewing ray through a
point is simply ``normalize(point)``. Inputs are therefore taken *before* the
pipeline's gravity alignment:

  hand    ``{raw}/{task}/all_hand_meshes.npz``      left_vertices (N,778,3)
  object  ``.../layout_camera_frame_optimized.json`` translation/quat_wxyz_camera_frame
  mesh    ``.../frame_*_masks/{obj}/{obj}.obj``      normalized; scaled by mesh_scale

The 1-D objective is fingertip-to-surface squared distance. That presumes the
hand's *articulation* is already right and only its placement is wrong — which
was verified for cupmove (a single rigid translation brings all five fingertips
to 0.1-0.4 cm, per-finger spread 0.30 cm). ``--report-anchors`` additionally
counts CHOIR-style contact anchors (within 2 cm, hand normal within a 60 deg
cone of the object direction) over *all* hand vertices, which is the quality
measure that does not presume which fingers touch.
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

# MANO fingertip vertex indices (778-vertex topology).
FINGERTIPS = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}
ANCHOR_MAX_DIST = 0.02      # CHOIR: correspondence kept only under 2 cm
ANCHOR_CONE_DEG = 60.0      # CHOIR: hand-normal / object-direction cone
# Every Nth hand vertex is checked for penetration during the scan. The penalty
# needs to know how much of the hand is buried, not precisely which parts.
_PEN_STRIDE = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="dataset/cupmove")
    p.add_argument("--task", default="cupmove")
    p.add_argument("--object", default="cup")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--mesh-scale", type=float, default=None,
                   help="Metric scale for the normalized SAM-3D mesh. Default: "
                        "read from the layout json's translation_scale_optimization.")
    p.add_argument("--scan-range", type=float, default=0.25,
                   help="Half-width (m) of the 1-D ray search.")
    p.add_argument("--scan-steps", type=int, default=501)
    p.add_argument("--surface-samples", type=int, default=60000,
                   help="Object surface points used for the distance queries.")
    p.add_argument("--report-anchors", action="store_true",
                   help="Also count CHOIR-style contact anchors before/after.")
    p.add_argument("--interaction-thresh", type=float, default=0.10,
                   help="A frame counts as an interaction frame when the raw hand "
                        "comes within this distance (m) of the object. Matches the "
                        "pipeline's hand_object_distance_thresh. CHOIR rectifies "
                        "only interaction frames; on approach/retreat frames the "
                        "hand is legitimately away and pulling it in is wrong.")
    p.add_argument("--force-frames", default=None, metavar="A:B",
                   help="Correct exactly these frames and no others, overriding "
                        "the automatic gate. For testing whether the correction "
                        "helps at all, separately from the harder question of "
                        "finding the grasp phase on its own.")
    p.add_argument("--min-raw-anchors", type=int, default=1,
                   help="Anchors the *uncorrected* hand must already have for a "
                        "frame to count as interaction. Guards the approach, "
                        "where the hand is legitimately away and correcting it "
                        "erases the reach the optimizer is supposed to track.")
    p.add_argument("--penetration-weight", type=float, default=1.0,
                   help="How much more a millimetre inside the object costs than "
                        "a millimetre away from it. The unsigned distance the scan "
                        "used before cannot tell the two apart and drifts inward.")
    p.add_argument("--out", default=None, help="Write per-frame z* to this .npz.")
    return p.parse_args()


def load_object_mesh(raw_dir: str, obj_name: str, scale: float) -> trimesh.Trimesh:
    cands = sorted(glob.glob(
        f"{raw_dir}/video_segmentation/masks/frame_*_masks/{obj_name}/{obj_name}.obj"
    ))
    if not cands:
        raise FileNotFoundError(f"no {obj_name}.obj under {raw_dir}/video_segmentation")
    mesh = trimesh.load(cands[0], force="mesh", process=False)
    mesh.apply_scale(scale)
    return mesh


def load_object_poses(raw_dir: str, obj_name: str, n_frames: int):
    path = (f"{raw_dir}/obj_tracking_out/{obj_name}/combined_visualization/"
            "layout_camera_frame_optimized.json")
    layout = json.load(open(path))
    trans = np.zeros((n_frames, 3))
    quat = np.zeros((n_frames, 4))
    quat[:, 0] = 1.0
    valid = np.zeros(n_frames, dtype=bool)
    for obj in layout["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if fi is None or not (0 <= fi < n_frames):
            continue
        ls = obj["local_to_scene"]
        t = np.asarray(ls["translation_camera_frame"], dtype=float)
        if t[2] <= 0:      # same negative-depth rejection as process_dataset.py
            continue
        trans[fi] = t
        quat[fi] = np.asarray(ls["quat_wxyz_camera_frame"], dtype=float)
        valid[fi] = True
    scale = layout.get("translation_scale_optimization", {}).get("mesh_scale")
    return trans, quat, valid, scale


def object_world(mesh: trimesh.Trimesh, t: np.ndarray, q_wxyz: np.ndarray):
    R = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
    return mesh.vertices @ R.T + t, R


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (unit length)."""
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-12)


def count_anchors(hand_v, hand_n, obj_pts, obj_tree) -> int:
    """CHOIR-style accepted correspondences: within 2 cm and inside a 60 deg cone."""
    d, j = obj_tree.query(hand_v)
    near = d < ANCHOR_MAX_DIST
    if not near.any():
        return 0
    direction = obj_pts[j[near]] - hand_v[near]
    dn = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(dn, 1e-12)
    cos = (direction * hand_n[near]).sum(1)
    return int((cos > np.cos(np.radians(ANCHOR_CONE_DEG))).sum())


def main() -> None:
    args = parse_args()
    raw = os.path.abspath(args.raw_dir)

    meshes = np.load(f"{raw}/{args.task}/all_hand_meshes.npz")
    hv = meshes[f"{args.side}_vertices"].astype(np.float64)   # (N, 778, 3) camera frame
    hf = meshes[f"{args.side}_faces"].astype(np.int64)
    n_frames = hv.shape[0]

    trans, quat, valid, layout_scale = load_object_poses(raw, args.object, n_frames)
    scale = args.mesh_scale if args.mesh_scale is not None else layout_scale
    if scale is None:
        raise SystemExit("mesh_scale not in layout json; pass --mesh-scale")
    mesh = load_object_mesh(raw, args.object, float(scale))

    print(f"frames={n_frames}  valid object poses={int(valid.sum())}  "
          f"mesh_scale={float(scale):.6f}  mesh bbox={np.round(mesh.extents, 4)} m")
    print(f"watertight={mesh.is_watertight}")

    tip_idx = np.array(list(FINGERTIPS.values()))
    ts = np.linspace(-args.scan_range, args.scan_range, args.scan_steps)
    zstar = np.full(n_frames, np.nan)

    force_lo = force_hi = None
    if args.force_frames:
        force_lo, force_hi = (int(x) for x in args.force_frames.split(":"))
        print("forcing correction on frames %d..%d only" % (force_lo, force_hi - 1))

    rows = []
    for f in range(n_frames):
        if not valid[f]:
            continue
        ow, _ = object_world(mesh, trans[f], quat[f])
        # Dense surface sampling: mesh vertices alone leave gaps on large faces.
        mesh_posed = trimesh.Trimesh(vertices=ow, faces=mesh.faces, process=False)
        samples, sample_face = trimesh.sample.sample_surface(
            mesh_posed, args.surface_samples,
        )
        tree = cKDTree(samples)
        # Outward normal at each sample, so the distance query can be signed.
        sample_n = mesh_posed.face_normals[sample_face]

        verts = hv[f]
        ray = verts.mean(0)
        ray = ray / np.linalg.norm(ray)          # camera at origin -> ray = direction to hand

        tips = verts[tip_idx]
        d_before = tree.query(tips)[0]

        # The distance to the surface is unsigned, so on its own it cannot tell
        # "touching" from "buried 4 mm in": both read as a small number, and
        # sliding slightly inside often reads *smaller*. Left that way the scan
        # drifts into the object — measured on this take, the corrected hand
        # penetrated the cup on every sampled frame (18 of 778 vertices, up to
        # 4.5 mm) where the raw hand mostly did not. The physics stage then has
        # to push the hand back out, and it leaves from wherever the contact
        # geometry sends it rather than from the demonstrated grasp.
        #
        # The sign comes from the outward normal at the nearest surface sample:
        # a point on the far side of it is inside. That reuses the query already
        # being made, where a ray-cast `contains` over the scan would dominate
        # the runtime.
        def signed(p):
            d, j = tree.query(p)
            inside = ((p - samples[j]) * sample_n[j]).sum(-1) < 0.0
            return np.where(inside, -d, d)

        # No z* both touches and stays outside: measured on frame 48, every
        # offset with a fingertip gap under 1.1 cm buries part of the hand, and
        # the only penetration-free offsets sit a centimetre or more away. A
        # rigid shift cannot fix a hand whose *shape* disagrees with the object
        # — the same limit that shows up as the 2.77 cm spread between fingers.
        # So this weight does not remove penetration, it only keeps the scan out
        # of the deeper wells (frame 48 has offsets burying 38 vertices against
        # the chosen 15). Raising it trades contact away instead.
        #
        # Attraction is judged on the fingertips, penetration on the whole hand.
        # They have to be measured on different point sets: the parts that end up
        # buried are knuckles and finger sides, not the tips, so a tips-only
        # penalty leaves the drift untouched — measured, z* moved 0.4 mm between
        # penalty weights of 1 and 50 when only the tips were charged.
        probe = verts[::_PEN_STRIDE]
        sd_tip = np.array([signed(tips + t * ray) for t in ts])       # (T, 5)
        sd_all = np.array([signed(probe + t * ray) for t in ts])      # (T, P)
        cost = (np.maximum(sd_tip, 0.0) ** 2).sum(1) + \
            args.penetration_weight * (np.maximum(-sd_all, 0.0) ** 2).sum(1)

        t_best = float(ts[int(cost.argmin())])
        d_after = tree.query(tips + t_best * ray)[0]
        zstar[f] = t_best

        # An interaction frame is one where the *raw* hand is already near the
        # object; elsewhere (approach, retreat, tracking dropout) the hand is
        # legitimately far and sliding it onto the surface is meaningless.
        raw_hand_gap = tree.query(verts)[0].min()
        # A frame is an interaction frame only if the *raw* hand already makes
        # contact-like correspondences. A distance threshold alone does not say
        # that: on this take the hand approaches from 5-6 cm, comfortably inside
        # any threshold wide enough to cover the grasp, with zero anchors — and
        # since the objective is "pull the fingertips onto the surface", feeding
        # it those frames pulls the hand onto the cup from frame 0. Measured, it
        # took the approach from a 4.63 cm gap to 0.15 cm and invented 86 anchors
        # a frame where the reconstruction had none. The demonstrated reach then
        # no longer exists for the optimizer to follow.
        hn_raw = vertex_normals(verts, hf)
        raw_anchors = count_anchors(verts, hn_raw, samples, tree)
        if force_lo is not None:
            interacting = bool(force_lo <= f < force_hi)
        else:
            interacting = bool(raw_hand_gap < args.interaction_thresh
                               and raw_anchors >= args.min_raw_anchors)
        # A z* pinned to the scan boundary means the search never found a
        # minimum -- treat it as a failed frame rather than a real correction.
        saturated = bool(abs(abs(t_best) - args.scan_range) < 1e-9)

        row = dict(frame=f, z=t_best,
                   before=d_before.min() * 100, after=d_after.min() * 100,
                   before_sum=d_before.sum() * 100, after_sum=d_after.sum() * 100,
                   spread=(d_after.max() - d_after.min()) * 100,
                   interacting=interacting, saturated=saturated,
                   raw_anchors=raw_anchors,
                   raw_gap=raw_hand_gap * 100)
        if args.report_anchors:
            hn = vertex_normals(verts, hf)
            row["anchors_before"] = count_anchors(verts, hn, samples, tree)
            row["anchors_after"] = count_anchors(verts + t_best * ray, hn, samples, tree)
        rows.append(row)

    print()
    hdr = ("frame |    z*(cm) | tip dist min (cm)   | tip dist sum (cm)   | spread")
    if args.report_anchors:
        hdr += " | anchors"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = ("%5d | %+8.2f  | %5.2f -> %5.2f      | %6.2f -> %6.2f     | %5.2f"
                % (r["frame"], r["z"] * 100, r["before"], r["after"],
                   r["before_sum"], r["after_sum"], r["spread"]))
        if args.report_anchors:
            line += " | %4d -> %4d" % (r["anchors_before"], r["anchors_after"])
        print(line)

    def report(sel, title):
        if not sel:
            print("\n=== %s ===\n  (no frames)" % title)
            return None
        b = np.array([r["before"] for r in sel]); a = np.array([r["after"] for r in sel])
        z = np.array([r["z"] for r in sel])
        print("\n=== %s: %d frames ===" % (title, len(sel)))
        print("  min tip-surface gap   before: mean %.2f cm  max %.2f cm  std %.2f"
              % (b.mean(), b.max(), b.std()))
        print("  min tip-surface gap   after : mean %.2f cm  max %.2f cm  std %.2f"
              % (a.mean(), a.max(), a.std()))
        print("  z*  mean %+.2f cm  std %.2f cm  range [%+.2f, %+.2f]"
              % (z.mean() * 100, z.std() * 100, z.min() * 100, z.max() * 100))
        if args.report_anchors:
            ab = np.array([r["anchors_before"] for r in sel])
            aa = np.array([r["anchors_after"] for r in sel])
            print("  anchors (<2cm & <60deg)  before: mean %.1f   after: mean %.1f"
                  % (ab.mean(), aa.mean()))
        return a

    if rows:
        report(rows, "ALL frames with a valid object pose")
        inter = [r for r in rows if r["interacting"] and not r["saturated"]]
        a = report(inter, "INTERACTION frames only (raw hand within %.0f cm)"
                   % (args.interaction_thresh * 100))
        skipped = [r for r in rows if not (r["interacting"] and not r["saturated"])]
        if skipped:
            print("\n  excluded %d frames: %s"
                  % (len(skipped), ", ".join(
                      "%d(%s)" % (r["frame"], "sat" if r["saturated"] else "far %.0fcm" % r["raw_gap"])
                      for r in skipped)))
        if a is not None:
            print("\n  GATE (interaction frames): 'after' gap under 0.5 cm ?  -> %s"
                  % ("PASS" if a.max() < 0.5 else "FAIL (max %.2f cm)" % a.max()))

    if args.out:
        def col(key, default=np.nan):
            a = np.full(n_frames, default, dtype=float)
            for r in rows:
                a[r["frame"]] = r[key]
            return a

        good = np.zeros(n_frames, dtype=bool)
        for r in rows:
            good[r["frame"]] = r["interacting"] and not r["saturated"]

        np.savez(
            args.out,
            z_star=zstar, valid=valid, good=good,
            frames=np.arange(n_frames), mesh_scale=float(scale),
            gap_before=col("before"), gap_after=col("after"),
            gap_sum_before=col("before_sum"), gap_sum_after=col("after_sum"),
            spread_after=col("spread"), raw_gap=col("raw_gap"),
            interacting=col("interacting", 0.0).astype(bool),
            saturated=col("saturated", 0.0).astype(bool),
            raw_anchors=col("raw_anchors", 0.0),
            anchors_before=col("anchors_before", 0.0) if args.report_anchors else np.zeros(n_frames),
            anchors_after=col("anchors_after", 0.0) if args.report_anchors else np.zeros(n_frames),
            interaction_thresh=float(args.interaction_thresh),
        )
        print(f"\nSaved per-frame z* + diagnostics -> {args.out}")


if __name__ == "__main__":
    main()
