"""Show the robot-side contact map next to the MANO one it came from.

  left    MANO hand, its contacting vertices coloured by weight
  right   the robot hand, the same contacts after the correspondence transfer

Colour runs grey -> red with weight, so the two panels should show the same
patches in the same places. What to look for: the coloured patches must sit on
the *palm side* of the fingers, and they must cover the middle phalanges and
not only the tips — that is how the demonstration actually holds the mug, with
the fingers hooked through the handle rather than pinching it.

    python viz_robot_contact_map.py
    python viz_robot_contact_map.py --color finger   # identity instead of weight
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


from viz_mano_robot_map import (  # noqa: E402
    FINGER_RGB, GREY, hand_mesh, spheres, to_view,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", default="outputs/robot_contact_map_left.npz")
    p.add_argument("--map", default="outputs/mano_robot_map_left.npz")
    p.add_argument("--scene", default=None)
    p.add_argument("--color", default="weight", choices=("weight", "finger"))
    p.add_argument("--sphere-radius", type=float, default=0.0026)
    p.add_argument("--out", default="outputs/robot_contact_map.glb")
    return p.parse_args()


def main() -> None:
    import mujoco

    args = parse_args()
    root = _package_root()
    c = np.load(args.contact, allow_pickle=True)
    m = np.load(args.map, allow_pickle=True)
    side = str(c["side"])
    scene = args.scene or str(c["scene"])

    from retargeting.utils.mano_robot_map import _bid, _canonical, _palmar_world
    mm = mujoco.MjModel.from_xml_path(
        str(root / f"retargeting/assets/robots/mano/{side}.xml"))
    rm = mujoco.MjModel.from_xml_path(scene)
    md, rd = _canonical(mm, side), _canonical(rm, side)

    vidx = c["mano_vertex"].astype(np.int64)
    w = np.asarray(c["weight"], float)
    body = c["body_id"].astype(np.int64)
    off = np.asarray(c["local_offset"], float)

    P = rd.xpos[body] + np.einsum(
        "nij,nj->ni", rd.xmat[body].reshape(-1, 3, 3), off)
    V = m["mano_template"][vidx]

    if args.color == "weight":
        lo, hi = np.array([170.0, 170.0, 170.0]), np.array([235.0, 40.0, 40.0])
        t = np.clip(w, 0, 1)[:, None]
        rgb = (lo * (1 - t) + hi * t).astype(np.uint8)
    else:
        rgb = np.array([FINGER_RGB[f] for f in c["finger"].astype(str)], np.uint8)
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1)

    mv, mf = hand_mesh(mm, md, side)
    rv, rf = hand_mesh(rm, rd, side)

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

    def place(p, R, o, dx):
        return to_view((p - o) @ R.T) + np.array([dx, 0.0, 0.0])

    width = max(np.ptp((mv - o_m) @ R_m.T, 0)[0],
                np.ptp((rv - o_r) @ R_r.T, 0)[0]) * 1.35

    out_scene = trimesh.Scene()
    for verts, faces, pts, R, o, dx in ((mv, mf, V, R_m, o_m, 0.0),
                                        (rv, rf, P, R_r, o_r, width)):
        h = trimesh.Trimesh(place(verts, R, o, dx), faces, process=False)
        h.visual = trimesh.visual.ColorVisuals(
            mesh=h, vertex_colors=np.tile(GREY, (len(verts), 1)))
        out_scene.add_geometry(h)
        out_scene.add_geometry(
            spheres(place(pts, R, o, dx), rgba, args.sphere_radius))

    out = Path(args.out)
    out_scene.export(out)
    print(f"Saved {out}   (left = MANO, right = {side} robot, colour = {args.color})")
    print(f"{len(w)} contact points   weight mean {w.mean():.3f}")


if __name__ == "__main__":
    main()
