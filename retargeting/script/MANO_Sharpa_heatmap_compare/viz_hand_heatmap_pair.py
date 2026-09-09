"""The hand-side contact map, on MANO and on the robot, side by side.

  left   MANO's 778 vertices shaded by ``hand_confidence`` from the extractor
  right  the robot's own skin shaded by the same weights after the transfer

Both are surface heatmaps rather than scattered markers, so the two are read
the same way and the question "did the patch survive the transfer" is a matter
of comparing two pictures of a hand.

The robot has no per-vertex confidence of its own — the transfer produces a
few hundred points, not a field — so its shading is those points splatted onto
the visual mesh with a Gaussian. ``--sigma`` sets that radius; it is a display
choice and changes nothing the reward sees.

    python viz_hand_heatmap_pair.py
    python viz_hand_heatmap_pair.py --max-points 64   # what the reward uses
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

def _package_root() -> Path:
    """Directory holding ``retargeting/``, wherever this script has been filed.

    These viewers used to sit next to the package and resolved assets relative
    to their own location. They have since been grouped by topic under
    ``script/``, so that assumption breaks; walking up until the package turns
    up keeps them working from either place, and from any working directory.
    """
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "retargeting" / "assets").is_dir():
            return p
    return here.parent


from viz_mano_robot_map import hand_mesh, to_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", default="../reconstruction/heatmap_out/contact_heatmap_palmar.npz")
    p.add_argument("--robot-contact", default="outputs/robot_contact_map_left.npz")
    p.add_argument("--map", default="outputs/mano_robot_map_left.npz")
    p.add_argument("--scene", default=None)
    p.add_argument("--sigma", type=float, default=0.008,
                   help="Gaussian radius (m) used to splat the robot's contact "
                        "points onto its mesh. Display only.")
    p.add_argument("--max-points", type=int, default=0,
                   help="Use only the heaviest N transferred points; 0 = all. "
                        "Set 64 to see exactly what the reward is given.")
    p.add_argument("--out", default="outputs/hand_heatmap_pair.glb")
    return p.parse_args()


def jet(t: np.ndarray) -> np.ndarray:
    """Blue -> cyan -> green -> yellow -> red, the usual heatmap reading."""
    t = np.clip(t, 0.0, 1.0)
    stops = np.array([[0.10, 0.20, 0.75], [0.10, 0.75, 0.85], [0.25, 0.80, 0.30],
                      [0.95, 0.85, 0.15], [0.85, 0.12, 0.12]])
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    return np.concatenate([rgb * 255, np.full((len(t), 1), 255.0)], 1).astype(np.uint8)


def main() -> None:
    import mujoco
    from retargeting.utils.mano_robot_map import _bid, _canonical, _palmar_world

    args = parse_args()
    root = _package_root()

    c = np.load(args.contact, allow_pickle=True)
    if "hand_confidence" not in c.files:
        raise SystemExit(f"{args.contact} has no hand_confidence; re-run "
                         "extract_contact_map.py")
    H_hand = np.clip(np.asarray(c["hand_confidence"], float), 0, 1)

    m = np.load(args.map, allow_pickle=True)
    side = str(m["side"])
    rc = np.load(args.robot_contact, allow_pickle=True)
    scene = args.scene or str(rc["scene"])
    if not Path(scene).is_absolute():
        scene = str(root / scene)

    mm = mujoco.MjModel.from_xml_path(
        str(root / f"retargeting/assets/robots/mano/{side}.xml"))
    rm = mujoco.MjModel.from_xml_path(scene)
    md, rd = _canonical(mm, side), _canonical(rm, side)

    body = rc["body_id"].astype(np.int64)
    off = np.asarray(rc["local_offset"], float)
    w = np.asarray(rc["weight"], float)
    if 0 < args.max_points < len(w):
        keep = np.argpartition(-w, args.max_points)[:args.max_points]
        body, off, w = body[keep], off[keep], w[keep]
    P = rd.xpos[body] + np.einsum("nij,nj->ni", rd.xmat[body].reshape(-1, 3, 3), off)

    mv, mf = hand_mesh(mm, md, side)
    rv, rf = hand_mesh(rm, rd, side)

    # MANO shades directly: its vertices are what the confidences index.
    vt = m["mano_template"]
    from scipy.spatial import cKDTree
    d_m, j_m = cKDTree(vt).query(mv)
    conf_m = np.where(d_m < 1e-4, H_hand[j_m], 0.0)

    # The robot has no such field, so splat the transferred points onto it.
    d2 = ((rv[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    conf_r = (w[None, :] * np.exp(-d2 / (2 * args.sigma ** 2))).sum(1)
    conf_r = conf_r / max(conf_r.max(), 1e-12)

    print(f"MANO  : {int((H_hand > 0).sum())} of {len(H_hand)} vertices in contact")
    print(f"robot : {len(P)} transferred points, splatted with sigma "
          f"{1000 * args.sigma:.0f} mm over {len(rv)} mesh vertices")

    def pose_align(model, data, palm_body, mcp_body):
        o = data.xpos[_bid(model, palm_body)]
        fwd = data.xpos[_bid(model, mcp_body)] - o
        fwd = fwd / np.linalg.norm(fwd)
        pal = _palmar_world(model, side, "middle")
        pal = pal - fwd * float(pal @ fwd)
        pal = pal / np.linalg.norm(pal)
        return np.stack([np.cross(fwd, pal), fwd, pal], axis=0), o

    R_m, o_m = pose_align(mm, md, f"{side}_palm", f"{side}_middle1z")
    R_r, o_r = pose_align(rm, rd, f"{side}_hand_C_MC", f"{side}_middle_MCP_VL")
    width = max(np.ptp((mv - o_m) @ R_m.T, 0)[0],
                np.ptp((rv - o_r) @ R_r.T, 0)[0]) * 1.3

    scene_out = trimesh.Scene()
    for verts, faces, conf, R, o, dx in ((mv, mf, conf_m, R_m, o_m, 0.0),
                                         (rv, rf, conf_r, R_r, o_r, width)):
        pts = to_view((verts - o) @ R.T) + np.array([dx, 0.0, 0.0])
        h = trimesh.Trimesh(pts, faces, process=False)
        h.visual = trimesh.visual.ColorVisuals(mesh=h, vertex_colors=jet(conf))
        scene_out.add_geometry(h)

    out = Path(args.out)
    scene_out.export(out)
    print(f"\nSaved {out}   (left = MANO, right = {side} robot; "
          "blue = untouched, red = most-touched)")


if __name__ == "__main__":
    main()
