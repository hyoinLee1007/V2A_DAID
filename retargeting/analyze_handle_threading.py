"""Is the finger *through* the handle, or just resting on it?

``analyze_grasp_contacts.py`` reports which face of a finger touches the cup.
That is a different question from the one the demonstration actually poses,
where the index and middle fingers pass through the handle's opening. A finger
laid against the outside of the handle can touch with its pad and score as a
perfectly good palmar contact while holding the mug in a way the demonstration
never does.

Threading is decided by what surrounds the finger. Take the handle vertices
near a fingertip and look at the directions from the tip to each of them: a
finger inside the loop has handle on all sides, so those directions span most
of a sphere. A finger resting outside has them bunched into one hemisphere.
The solid angle they cover separates the two without needing the mesh to be
watertight or the loop to be found explicitly.

Also reported is whether the mug stays upright, since a run can produce good
contact numbers while tipping the cup over.

    python analyze_handle_threading.py --run outputs/sharpa/left/cupmove/0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Handle vertices are picked out by standing off the mug's axis of revolution.
_HANDLE_RADIUS_FACTOR = 1.35
# Handle vertices within this of a fingertip count as surrounding it.
_NEAR = 0.03
# Fraction of directions that must lie opposite the dominant one to call it
# threaded. A finger in the loop sees handle both in front and behind.
_THREAD_FRAC = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/sharpa/left/cupmove/0")
    p.add_argument("--scene", default=None,
                   help="scene.xml to use (archived copies lose their asset paths).")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--stride", type=int, default=10)
    return p.parse_args()


def handle_mask(verts: np.ndarray) -> np.ndarray:
    """Vertices standing off the mug body — the handle.

    The mug body is a surface of revolution; its axis is the direction whose
    slices have the most constant radius. Anything well outside the local body
    radius at its own height is handle.
    """
    c = verts.mean(0)
    X = verts - c

    def spread(a):
        a = a / np.linalg.norm(a)
        t = X @ a
        r = np.linalg.norm(X - np.outer(t, a), axis=1)
        b = np.clip(((t - t.min()) / (np.ptp(t) + 1e-9) * 20).astype(int), 0, 19)
        s = [np.median(np.abs(r[b == i] - np.median(r[b == i]))) / max(np.median(r[b == i]), 1e-9)
             for i in range(20) if (b == i).sum() > 30]
        return float(np.mean(s))

    axis = min((spread(np.random.default_rng(i).normal(size=3)),
                np.random.default_rng(i).normal(size=3)) for i in range(400))[1]
    axis = axis / np.linalg.norm(axis)
    t = X @ axis
    r = np.linalg.norm(X - np.outer(t, axis), axis=1)
    b = np.clip(((t - t.min()) / (np.ptp(t) + 1e-9) * 40).astype(int), 0, 39)
    body = np.array([np.median(r[b == i]) if (b == i).sum() > 10 else np.nan
                     for i in range(40)])[b]
    return (r > body * _HANDLE_RADIUS_FACTOR) & ~np.isnan(body)


def threaded(tip: np.ndarray, handle_pts: np.ndarray) -> bool:
    """True when handle surrounds the tip rather than sitting to one side."""
    d = handle_pts - tip
    n = np.linalg.norm(d, axis=1)
    near = n < _NEAR
    if near.sum() < 8:
        return False
    u = d[near] / n[near, None]
    dominant = u.mean(0)
    dominant /= max(np.linalg.norm(dominant), 1e-9)
    return float((u @ dominant < 0.0).mean()) > _THREAD_FRAC


def main() -> None:
    import mujoco
    from retargeting.utils.mano_robot_map import _bid

    args = parse_args()
    run = Path(args.run)
    side = args.side
    model = mujoco.MjModel.from_xml_path(args.scene or str(run / "scene.xml"))
    data = mujoco.MjData(model)
    qpos = np.asarray(np.load(run / "trajectory_mjwp.npz",
                              allow_pickle=True)["qpos"], float).reshape(-1, model.nq)

    # In the object BODY frame, not raw mesh coordinates. Every mesh geom in
    # this scene carries a non-identity geom_pos/geom_quat, and left_visual's is
    # a 132.1 deg rotation with an 18.9 mm offset (156.4 mm of per-vertex
    # difference). Posing raw mesh_vert with the body transform put the handle
    # 132 deg away from where it actually is, so every threading test since this
    # tool was written has been measuring a phantom handle — which is why the
    # answer was 0/120 in every run regardless of what changed.
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    V0 = _visual_mesh_local_verts(model, side)
    nv = len(V0)
    hm = handle_mask(V0)
    ob = _bid(model, f"{side}_object")
    # The mug's own up axis, read off its rest pose in the scene.
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    up_local = np.array([0.0, 1.0, 0.0])

    fingers = ("thumb", "index", "middle", "ring", "pinky")
    tips = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{f}_tip")
            for f in fingers}

    counts = {f: 0 for f in fingers}
    tilt = []
    n = 0
    for s in range(0, len(qpos), args.stride):
        data.qpos[:] = qpos[s]
        mujoco.mj_forward(model, data)
        R = data.xmat[ob].reshape(3, 3)
        hp = V0[hm] @ R.T + data.xpos[ob]
        tilt.append(np.degrees(np.arccos(np.clip((R @ up_local) @ [0, 0, 1.0], -1, 1))))
        for f in fingers:
            if threaded(data.site_xpos[tips[f]], hp):
                counts[f] += 1
        n += 1

    tilt = np.asarray(tilt)
    print(f"handle vertices: {int(hm.sum())} of {nv}")
    print(f"frames sampled : {n}\n")
    print("fingertip inside the handle loop:")
    for f in fingers:
        print(f"  {f:7s} {counts[f]:4d} / {n}  ({100 * counts[f] / max(n, 1):5.1f}% of frames)")
    print(f"\nmug tilt from upright:  start {tilt[0]:5.1f} deg   "
          f"end {tilt[-1]:5.1f} deg   max {tilt.max():5.1f} deg")
    if tilt[-1] > 45:
        print("  the mug ends up on its side — the grasp did not hold it")


if __name__ == "__main__":
    main()
