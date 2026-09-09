"""Build the MANO-vertex -> robot-hand correspondence table.

Writes an npz holding, for each of MANO's 778 vertices, the robot body it maps
to and the offset within that body's frame. Evaluating

    xpos[body] + xmat[body] @ local_offset

on the live simulation then gives the robot-side point for any CHOIR anchor,
with no change to the robot XML. See ``retargeting/utils/mano_robot_map.py``
for how the correspondence is derived and why it is built this way.

    python build_mano_robot_map.py \
        --scene outputs/sharpa/left/cupmove/0/scene.xml --side left

Validation printed at the end:

  snap distance      how far the transferred point had to move to reach the
                     robot's skin. Large values mean the two segments disagree
                     in shape beyond what one anisotropic scale can absorb.
  side agreement     fraction of vertices that stayed on the palmar side they
                     started on. This is the number that matters: the whole
                     purpose of the table is to keep pad and back distinct.
  fingertip check    MANO's five tip vertices should land near the robot's own
                     ``*_tip`` sites, which were placed independently of any of
                     this and so make a free consistency test.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retargeting.utils.mano_robot_map import (  # noqa: E402
    _bid, _canonical, _palmar_world, _sid, _surface_mesh_world, _surface_world,
    _denormalise, _normalise, _radial, segment_frame, segments, snap_to_surface,
)

# MANO's conventional fingertip vertex indices, used only as a check.
TIP_VERT = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="Robot scene.xml to map onto.")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--mano-xml", default=None,
                   help="MANO MJCF (default: assets/robots/mano/<side>.xml).")
    p.add_argument("--mano-pkl", default=None,
                   help="MANO_{LEFT,RIGHT}.pkl, for the 778-vertex template.")
    p.add_argument("--out", default="outputs/mano_robot_map_{side}.npz")
    return p.parse_args()


def load_mano_template(path: str) -> np.ndarray:
    """``v_template`` in the MANO MJCF's frame (i.e. shifted to the wrist).

    The pkl is a pickled chumpy graph; a shim standing in for chumpy is enough
    to read the numeric leaves, and avoids adding a dependency that no longer
    installs cleanly on this Python.
    """
    class _Node:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, st):
            self.__dict__.update(st if isinstance(st, dict) else {"_s": st})

    def _attr(n):
        # Dunders must keep failing: torch introspects __file__ on every loaded
        # module, and handing it a class breaks unrelated imports.
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return type(n, (_Node,), {})

    saved = {n: sys.modules.get(n) for n in
             ("chumpy", "chumpy.ch", "chumpy.reordering", "chumpy.optimization",
              "chumpy.linalg", "chumpy.utils", "chumpy.logic")}
    try:
        for name in saved:
            mod = types.ModuleType(name)
            mod.__getattr__ = _attr
            sys.modules[name] = mod
        with open(path, "rb") as fh:
            m = pickle.load(fh, encoding="latin1")
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return np.asarray(m["v_template"], float) - np.asarray(m["J"], float)[0]


def main() -> None:
    import mujoco

    args = parse_args()
    side = args.side
    root = Path(__file__).resolve().parent
    mano_xml = args.mano_xml or root / f"retargeting/assets/robots/mano/{side}.xml"
    mano_pkl = args.mano_pkl or (
        root.parent / "reconstruction/modules/HaWoR/_DATA"
        / (f"data_left/mano_left/MANO_LEFT.pkl" if side == "left"
           else "data/mano/MANO_RIGHT.pkl"))

    mm = mujoco.MjModel.from_xml_path(str(mano_xml))
    rm = mujoco.MjModel.from_xml_path(str(args.scene))
    md, rd = _canonical(mm, side), _canonical(rm, side)

    vt = load_mano_template(str(mano_pkl))
    n_v = len(vt)
    print(f"MANO template: {n_v} vertices   robot: {rm.nbody} bodies\n")

    # Which MANO vertices belong to which MJCF part. The part meshes reproduce
    # the template exactly, so a nearest-neighbour lookup is an exact identity
    # here, not an approximation — the printed residual confirms it.
    from scipy.spatial import cKDTree
    part_pts, part_owner = [], []
    for b in range(mm.nbody):
        nm = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        has_mesh = any(int(mm.geom_bodyid[g]) == b
                       and int(mm.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                       and int(mm.geom_dataid[g]) >= 0 for g in range(mm.ngeom))
        if not has_mesh:
            continue
        v = _surface_world(mm, md, [nm])
        part_pts.append(v)
        part_owner += [nm] * len(v)
    part_pts = np.concatenate(part_pts)
    part_owner = np.array(part_owner)
    d_id, j_id = cKDTree(part_pts).query(vt)
    print(f"template <-> part-mesh identity residual: max {d_id.max():.2e} m")
    if d_id.max() > 1e-4:
        raise SystemExit("MANO MJCF part meshes do not match the template; "
                         "the segmentation would be guesswork.")
    vert_body = part_owner[j_id]

    body_id = np.full(n_v, -1, np.int64)
    offset = np.zeros((n_v, 3))
    seg_of = np.full(n_v, "", dtype=object)
    finger_of = np.full(n_v, "", dtype=object)
    snap_d = np.full(n_v, np.nan)
    side_ok = np.zeros(n_v, bool)
    # Palmar coordinate of each MANO vertex in its own segment frame: +1 at the
    # palm-side skin, -1 at the back. Purely a MANO-side quantity — no robot is
    # involved — but it is free to compute here and is what lets a contact
    # extractor keep palm-side correspondences only.
    mano_palmar = np.zeros(n_v)

    print(f"\n{'segment':9s} {'verts':>5s} {'sx m/m':>13s} {'sy':>11s} {'sz':>11s} "
          f"{'snap mm':>16s} {'side':>6s}")
    print("-" * 80)

    for seg in segments(side):
        sel = np.flatnonzero(vert_body == seg["mano_frame"])
        if len(sel) == 0:
            print(f"{seg['name']:9s}     0   (no MANO vertices)")
            continue

        palm_m = _palmar_world(mm, side, seg["finger"])
        palm_r = _palmar_world(rm, side, seg["finger"])
        surf_m = _surface_world(mm, md, seg["mano_surf"])
        surf_r, faces_r = _surface_mesh_world(rm, rd, seg["rob_surf"])

        fr_m = segment_frame(mm, md, seg["mano_frame"], seg["mano_distal"],
                             palm_m, surf_m)
        fr_r = segment_frame(rm, rd, seg["rob_frame"], seg["rob_distal"],
                             palm_r, surf_r)

        u = _normalise(vt[sel], fr_m)
        p = _denormalise(u, fr_r)
        p_snap = snap_to_surface(p, surf_r, fr_r, faces_r)

        # Did the palmar/dorsal side survive the transfer and the snap?
        z_m = _normalise(vt[sel], fr_m)[:, 2]
        z_r = _normalise(p_snap, fr_r)[:, 2]
        ok = np.sign(z_m) == np.sign(z_r)

        b = _bid(rm, seg["rob_frame"])
        R = rd.xmat[b].reshape(3, 3)
        body_id[sel] = b
        offset[sel] = (p_snap - rd.xpos[b]) @ R
        seg_of[sel] = seg["name"]
        finger_of[sel] = seg["finger"] if seg["name"] != "palm" else "palm"
        snap_d[sel] = np.linalg.norm(p_snap - p, axis=1)
        side_ok[sel] = ok
        mano_palmar[sel] = z_m

        print(f"{seg['name']:9s} {len(sel):5d} "
              f"{fr_m['sx']:.4f}->{fr_r['sx']:.4f} "
              f"{fr_m['sy']:.4f}->{fr_r['sy']:.4f} "
              f"{fr_m['sz']:.4f}->{fr_r['sz']:.4f} "
              f"{1000 * snap_d[sel].mean():6.1f} (max {1000 * snap_d[sel].max():5.1f}) "
              f"{100 * ok.mean():5.1f}%")

    unmapped = int((body_id < 0).sum())
    print("-" * 80)
    print(f"mapped {n_v - unmapped}/{n_v} vertices"
          + (f"   UNMAPPED {unmapped}" if unmapped else ""))
    print(f"snap distance: mean {1000 * np.nanmean(snap_d):.1f} mm  "
          f"median {1000 * np.nanmedian(snap_d):.1f} mm  "
          f"p95 {1000 * np.nanpercentile(snap_d, 95):.1f} mm")
    print(f"side agreement: {100 * side_ok.mean():.1f}%")

    print("\nfingertip check: MANO's tip vertex should reach the robot's actual")
    print("fingertip. That is the distal end of the distal link's skin, not the")
    print(f"{side}_*_tip site, which sits mid-phalanx on this robot.")
    from retargeting.utils.mano_robot_map import eval_points
    P = eval_points(body_id.clip(0), offset, rd.xpos[None], rd.xmat[None])[0]
    for f, vi in TIP_VERT.items():
        # The robot fingertip is the skin vertex farthest along the link's bone.
        surf = _surface_world(rm, rd, [f"{side}_{f}_DP"])
        o = rd.xpos[_bid(rm, f"{side}_{f}_DP")]
        ax = rd.site_xpos[_sid(rm, f"{side}_{f}_tip")] - o
        ax = ax / np.linalg.norm(ax)
        far = surf[int(np.argmax((surf - o) @ ax))]
        site = rd.site_xpos[_sid(rm, f"{side}_{f}_tip")]
        print(f"  {f:7s} MANO v{vi:4d} -> {seg_of[vi]:8s}   "
              f"to robot fingertip {100 * np.linalg.norm(P[vi] - far):5.2f} cm"
              f"   (to {f}_tip site {100 * np.linalg.norm(P[vi] - site):.2f} cm)")

    out = Path(args.out.format(side=side))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             body_id=body_id, local_offset=offset.astype(np.float64),
             segment=seg_of.astype(str), finger=finger_of.astype(str),
             snap_dist=snap_d, side_ok=side_ok, mano_palmar=mano_palmar,
             body_name=np.array([mujoco.mj_id2name(rm, mujoco.mjtObj.mjOBJ_BODY, int(b))
                                 if b >= 0 else "" for b in body_id]),
             mano_template=vt, side=side, scene=str(args.scene))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
