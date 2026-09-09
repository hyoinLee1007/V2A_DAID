"""Export the MANO -> robot correspondence as a GLB you can orbit and check.

Two panels side by side at the open-hand pose:

  left    MANO's 778 vertices, drawn on the MANO mesh
  right   the robot hand in grey, with each MANO vertex drawn as a sphere at
          the place the table maps it to

Both panels use the same colours, so the check is simply whether a patch keeps
its colour and its place: the index pad on the left should be the index pad on
the right, not the index back and not the middle finger.

    python viz_mano_robot_map.py --color finger    # which finger  (default)
    python viz_mano_robot_map.py --color side      # palmar vs dorsal
    python viz_mano_robot_map.py --color segment   # which phalanx

``--color side`` is the strict one. Green is the palm side, red the back,
taken from the sign of the vertex's palmar coordinate *on the robot* after the
transfer. Any red inside a region that is green on the left is a vertex whose
side did not survive, which is the one error this table exists to prevent.
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


FINGER_RGB = {
    "thumb":  (222, 60, 60),
    "index":  (245, 150, 40),
    "middle": (215, 205, 45),
    "ring":   (60, 150, 235),
    "pinky":  (165, 90, 215),
    "palm":   (140, 150, 160),
}
GREY = np.array([175, 175, 175, 255], np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="outputs/mano_robot_map_left.npz")
    p.add_argument("--scene", default=None,
                   help="Robot scene.xml (default: the one recorded in the map).")
    p.add_argument("--mano-xml", default=None)
    p.add_argument("--color", default="finger", choices=("finger", "side", "segment"))
    p.add_argument("--sphere-radius", type=float, default=0.0018)
    p.add_argument("--out", default="outputs/mano_robot_map.glb")
    return p.parse_args()


def body_mesh_world(model, data, body_id: int):
    """Visual-mesh vertices and triangles of one body, in world coords."""
    import mujoco
    V, F, base = [], [], 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body_id:
            continue
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        v = np.asarray(model.mesh_vert[va:va + vn], float)
        f = np.asarray(model.mesh_face[fa:fa + fn], np.int64)
        from retargeting.utils.mujoco_utils import quat_wxyz_to_rotmat
        v = v @ quat_wxyz_to_rotmat(model.geom_quat[g]).T + model.geom_pos[g]
        v = v @ data.xmat[body_id].reshape(3, 3).T + data.xpos[body_id]
        V.append(v)
        F.append(f + base)
        base += vn
    if not V:
        return None, None
    return np.concatenate(V), np.concatenate(F)


def hand_mesh(model, data, side: str):
    """Every visual mesh belonging to this hand, in world coords.

    A scene.xml holds the manipulated object under ``{side}_object``, so
    selecting bodies by the side prefix alone welds the cup onto the wrist.
    Only the palm and the five finger chains are hand.
    """
    import mujoco
    from retargeting.utils.mano_robot_map import _FINGERS
    keep = tuple(f"{side}_{f}" for f in _FINGERS) + (f"{side}_palm", f"{side}_hand")
    V, F, base = [], [], 0
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not nm.startswith(keep):
            continue
        v, f = body_mesh_world(model, data, b)
        if v is None:
            continue
        V.append(v)
        F.append(f + base)
        base += len(v)
    if not V:
        raise SystemExit(f"no hand meshes matched {keep}")
    return np.concatenate(V), np.concatenate(F)


# MuJoCo scenes are z-up; glTF viewers are y-up. Without this the hands arrive
# lying on their side and orbiting fights you.
_VIEW_FIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def to_view(p: np.ndarray) -> np.ndarray:
    return p @ _VIEW_FIX.T


def spheres(centers, colors, radius):
    base = trimesh.creation.icosphere(radius=radius, subdivisions=1)
    nb = len(base.vertices)
    v = np.tile(base.vertices, (len(centers), 1)) + np.repeat(centers, nb, 0)
    f = np.concatenate([base.faces + i * nb for i in range(len(centers))])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.repeat(colors, nb, 0))
    return m


def main() -> None:
    import mujoco

    args = parse_args()
    root = _package_root()
    d = np.load(args.map, allow_pickle=True)
    side = str(d["side"])
    scene = args.scene or str(d["scene"])
    mano_xml = args.mano_xml or root / f"retargeting/assets/robots/mano/{side}.xml"

    from retargeting.utils.mano_robot_map import (
        _bid, _canonical, _palmar_world, eval_points)
    mm = mujoco.MjModel.from_xml_path(str(mano_xml))
    rm = mujoco.MjModel.from_xml_path(scene)
    md, rd = _canonical(mm, side), _canonical(rm, side)

    body_id = d["body_id"].astype(np.int64)
    P = eval_points(body_id.clip(0), d["local_offset"], rd.xpos[None], rd.xmat[None])[0]
    vt = d["mano_template"]

    if args.color == "finger":
        rgb = np.array([FINGER_RGB[f] for f in d["finger"]], np.uint8)
    elif args.color == "side":
        ok = d["side_ok"].astype(bool)
        rgb = np.where(ok[:, None], np.array([[60, 200, 90]]),
                       np.array([[235, 45, 45]])).astype(np.uint8)
        print(f"side agreement {100 * ok.mean():.1f}%  "
              f"({int((~ok).sum())} vertices flipped)")
    else:
        segs = sorted(set(d["segment"].tolist()))
        cmap = {s: tuple(int(255 * c) for c in
                         trimesh.visual.color.hsv_to_rgba(
                             [[i / len(segs), 0.75, 0.95, 1.0]])[0][:3])
                for i, s in enumerate(segs)}
        rgb = np.array([cmap[s] for s in d["segment"]], np.uint8)
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1)

    mv, mf = hand_mesh(mm, md, side)
    rv, rf = hand_mesh(rm, rd, side)

    def pose_align(model, data, palm_body: str, mcp_body: str):
        """Rotation putting a hand into a shared display orientation.

        The two models are authored along different axes — MANO's fingers run
        down +x, sharpa's down +z — so shown as-authored they sit at ninety
        degrees to each other and cannot be compared by eye. Both are rebuilt
        onto their own anatomy instead: fingers up the screen, palm toward the
        viewer.
        """
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

    scene_out = trimesh.Scene()

    mano = trimesh.Trimesh(place(mv, R_m, o_m, 0.0), mf, process=False)
    mano.visual = trimesh.visual.ColorVisuals(
        mesh=mano, vertex_colors=np.tile([215, 215, 215, 90], (len(mv), 1)).astype(np.uint8))
    scene_out.add_geometry(mano)
    scene_out.add_geometry(spheres(place(vt, R_m, o_m, 0.0), rgba, args.sphere_radius))

    rob = trimesh.Trimesh(place(rv, R_r, o_r, width), rf, process=False)
    rob.visual = trimesh.visual.ColorVisuals(
        mesh=rob, vertex_colors=np.tile(GREY, (len(rv), 1)))
    scene_out.add_geometry(rob)
    scene_out.add_geometry(spheres(place(P, R_r, o_r, width), rgba, args.sphere_radius))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene_out.export(out)
    print(f"Saved {out}   (left = MANO, right = {side} robot, colour = {args.color})")
    if args.color == "finger":
        print("legend: " + "  ".join(f"{k}" for k in FINGER_RGB))


if __name__ == "__main__":
    main()
