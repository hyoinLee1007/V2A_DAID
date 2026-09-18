"""Draw the reward's targets: which patch of robot skin is meant to touch where.

The contact map is 64 points on the robot's hand, each carrying the object
vertices it met in the demonstration. Every number this project reports about
"distance to its own target" is that pairing, and a table of centimetres does
not show which point is off or in which direction.

At one step this draws, in the cup's frame:

  cup + robot hand      grey
  skin point            sphere, coloured by finger
  its target patch      spheres of the same colour on the cup
  the gap               a stick between the point and its nearest own target

so a finger whose sticks are long, or point somewhere unexpected, is visible
rather than inferred.

    python viz_target_pairs.py --step 900 --key qpos_ref_raw
    python viz_target_pairs.py --step 900 --key qpos          # optimized result
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "script" / "MANO_Sharpa_heatmap_compare"))

FINGER_RGB = {"thumb": (232, 92, 60), "index": (54, 126, 232),
              "middle": (46, 176, 106), "ring": (196, 110, 220),
              "pinky": (232, 176, 50), "palm": (130, 138, 150)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/sharpa/left/cupmove_decay2/0")
    p.add_argument("--scene", default=None)
    p.add_argument("--map", default="outputs/robot_contact_map_left.npz")
    p.add_argument("--side", default="left")
    p.add_argument("--key", default="qpos_ref_raw",
                   help="qpos_ref_raw / qpos_ref / qpos")
    p.add_argument("--step", type=int, default=900)
    p.add_argument("--all-targets", action="store_true",
                   help="Draw every target vertex, not just the nearest one.")
    p.add_argument("--radius", type=float, default=0.0025)
    p.add_argument("--stick", type=float, default=0.0009)
    p.add_argument("--compare", default=None,
                   help="Second pose (another --key value) drawn beside the first "
                        "in the same cup frame, so the two are comparable without "
                        "matching camera angles by hand.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def spheres(c, col, r):
    b = trimesh.creation.icosphere(radius=r, subdivisions=1)
    nb = len(b.vertices)
    v = np.tile(b.vertices, (len(c), 1)) + np.repeat(c, nb, 0)
    f = np.concatenate([b.faces + i * nb for i in range(len(c))])
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.repeat(col, nb, 0))
    return m


def sticks(a, b, col, r):
    u = trimesh.creation.cylinder(radius=r, height=1.0, sections=6)
    uv, uf = np.asarray(u.vertices), np.asarray(u.faces)
    nb = len(uv)
    d = b - a
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-6
    a, b, d, L, col = a[ok], b[ok], d[ok], L[ok], col[ok]
    z = d / L[:, None]
    tmp = np.tile([0.0, 0.0, 1.0], (len(z), 1))
    tmp[np.abs((z * tmp).sum(1)) > 0.9] = [1.0, 0.0, 0.0]
    x = np.cross(tmp, z); x /= np.linalg.norm(x, axis=1, keepdims=True)
    R = np.stack([x, np.cross(z, x), z], axis=-1)
    s = np.tile(uv, (len(z), 1, 1)); s[:, :, 2] *= L[:, None]
    v = np.einsum("nij,nvj->nvi", R, s) + ((a + b) / 2)[:, None, :]
    m = trimesh.Trimesh(v.reshape(-1, 3),
                        np.concatenate([uf + i * nb for i in range(len(z))]), process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.repeat(col, nb, 0))
    return m


def coloured(v, f, rgba):
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=np.tile(rgba, (len(v), 1)))
    return m


def main():
    import mujoco
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    from retargeting.utils.paired_contact_cost import build_targets
    from retargeting.utils.mano_robot_map import _bid
    from viz_mano_robot_map import hand_mesh

    a = parse_args()
    run = Path(a.run)
    m = mujoco.MjModel.from_xml_path(a.scene or str(run / "scene.xml"))
    d = mujoco.MjData(m)
    ob = _bid(m, f"{a.side}_object")

    cm = np.load(a.map, allow_pickle=True)
    w = np.asarray(cm["weight"], float)
    keep = np.argpartition(-w, 64)[:64] if len(w) > 64 else np.arange(len(w))
    body = cm["body_id"].astype(np.int64)[keep]
    off = np.asarray(cm["local_offset"], float)[keep]
    fing = cm["finger"].astype(str)[keep]
    V = _visual_mesh_local_verts(m, a.side)
    tx, to, te = build_targets(cm["tgt_offset"], cm["tgt_vertex"], cm["tgt_weight"],
                               keep, V, 0.0, "cpu", torch.float64)

    if a.key.endswith(".npz"):
        # A saved pose from probe_grasp_feasibility.py rather than a trajectory.
        q = np.asarray(np.load(a.key)["qpos"], float).reshape(m.nq)
    else:
        t = np.load(run / "trajectory_mjwp.npz", allow_pickle=True)
        q = np.asarray(t[a.key], float).reshape(-1, m.nq)[a.step]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)

    Rc = d.xmat[ob].reshape(3, 3)
    tc = d.xpos[ob].copy()
    cup = lambda p: (p - tc) @ Rc                                    # noqa: E731

    P = d.xpos[body] + np.einsum("nij,nj->ni", d.xmat[body].reshape(-1, 3, 3), off)
    Pl = cup(P)
    Tl = tx.numpy()                                                  # (K, T, 3), cup frame
    dist = np.linalg.norm(Pl[:, None, :] - Tl, axis=-1) + to.numpy()
    j = dist.argmin(1)
    near = Tl[np.arange(len(j)), j]
    gap = np.linalg.norm(Pl - near, axis=1)

    rgb = np.array([FINGER_RGB.get(f, (140, 140, 140)) for f in fing], float)
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255.0)], 1).astype(np.uint8)

    rv, rf = hand_mesh(m, d, a.side)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, f"{a.side}_visual")
    fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    cf = np.asarray(m.mesh_face[fa:fa + nf], np.int64)

    sc = trimesh.Scene()
    sc.add_geometry(coloured(V, cf, np.array([196, 176, 148, 255], np.uint8)))
    sc.add_geometry(coloured(cup(rv), rf, np.array([225, 225, 230, 255], np.uint8)))
    sc.add_geometry(spheres(Pl, rgba, a.radius))
    if a.all_targets:
        keepmask = to.numpy() < 1e2
        allc = np.concatenate([Tl[i][keepmask[i]] for i in range(len(Tl))])
        allr = np.concatenate([np.tile(rgba[i], (int(keepmask[i].sum()), 1))
                               for i in range(len(Tl))])
        sc.add_geometry(spheres(allc, allr, a.radius * 0.55))
    else:
        sc.add_geometry(spheres(near, rgba, a.radius * 0.8))
    sc.add_geometry(sticks(Pl, near, rgba, a.stick))

    if a.compare:
        q2 = (np.asarray(np.load(a.compare)["qpos"], float).reshape(m.nq)
              if a.compare.endswith(".npz")
              else np.asarray(np.load(run / "trajectory_mjwp.npz",
                                      allow_pickle=True)[a.compare],
                              float).reshape(-1, m.nq)[a.step])
        d.qpos[:] = q2
        mujoco.mj_forward(m, d)
        R2 = d.xmat[ob].reshape(3, 3)
        t2 = d.xpos[ob].copy()
        cup2 = lambda p: (p - t2) @ R2                               # noqa: E731
        P2 = d.xpos[body] + np.einsum("nij,nj->ni", d.xmat[body].reshape(-1, 3, 3), off)
        P2l = cup2(P2)
        d2 = np.linalg.norm(P2l[:, None, :] - Tl, axis=-1) + to.numpy()
        j2 = d2.argmin(1)
        near2 = Tl[np.arange(len(j2)), j2]
        rv2, rf2 = hand_mesh(m, d, a.side)
        # Offset along x by the pair's own width so the two do not overlap.
        span = np.ptp(np.vstack([V, cup(rv)]), 0)[0] * 1.35
        sh = np.array([span, 0.0, 0.0])
        sc.add_geometry(coloured(V + sh, cf, np.array([196, 176, 148, 255], np.uint8)))
        sc.add_geometry(coloured(cup2(rv2) + sh, rf2,
                                 np.array([225, 225, 230, 255], np.uint8)))
        sc.add_geometry(spheres(P2l + sh, rgba, a.radius))
        sc.add_geometry(spheres(near2 + sh, rgba, a.radius * 0.8))
        sc.add_geometry(sticks(P2l + sh, near2 + sh, rgba, a.stick))
        g2 = np.linalg.norm(P2l - near2, axis=1)
        print(f"\n비교 대상: {a.compare}")
        for f in sorted(set(fing)):
            k = fing == f
            print(f"  {f:8s} {g2[k].mean()*100:7.2f} cm  <1cm {int((g2[k]<0.01).sum()):3d}")

    out = a.out or (f"outputs/target_pairs_{Path(a.key).stem}.glb"
                    if a.key.endswith(".npz")
                    else f"outputs/target_pairs_{a.key}_{a.step}.glb")
    sc.export(out)
    print(f"{a.key}[{a.step}]   {len(P)} skin points")
    print(f"{'finger':8s} {'pts':>4s} {'gap mean':>10s} {'min':>8s} {'<1cm':>6s}")
    for f in sorted(set(fing)):
        k = fing == f
        print(f"{f:8s} {int(k.sum()):4d} {gap[k].mean()*100:7.2f} cm "
              f"{gap[k].min()*100:5.2f} cm {int((gap[k]<0.01).sum()):6d}")
    print(f"\nSaved {out}   (색 = 손가락, 막대 = 스킨점 → 자기 목표까지)")


if __name__ == "__main__":
    main()
