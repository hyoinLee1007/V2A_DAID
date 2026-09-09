"""Measure where the optimized grasp actually touches the object.

The reward terms tell you what the optimizer was scored on; they do not tell
you what it did. This replays the saved trajectory and looks at the contacts
themselves, answering the two questions the whole robot-skin term exists for:

  which links   is the object held by the fingers the demonstration used?
  which face    is it held by their pads, or by their backs?

The palmar coordinate reuses the segment frames from ``mano_robot_map``: a
contact is written in its link's own frame and read off along the palmar axis,
so +1 is the pad, -1 the back, and the sign is the answer.

A ``repel`` of zero in the reward log is ambiguous on its own — no contact away
from the mapped patches, or no contact at all. The contact count here settles
which.

    python analyze_grasp_contacts.py --run outputs/sharpa/left/cupmove/0
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/sharpa/left/cupmove/0")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--map", default="outputs/robot_contact_map_left.npz")
    p.add_argument("--from-step", type=int, default=None,
                   help="First step to analyse (default: end of warmup).")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--scene", default=None,
                   help="scene.xml to use. Archived runs keep a copy whose mesh "
                        "paths are relative and no longer resolve; point this at "
                        "a live run's scene.xml, which is the same geometry.")
    p.add_argument("--reference", action="store_true",
                   help="Score the reference the optimizer tracked (qpos_ref in "
                        "trajectory_mjwp.npz) instead of what it produced. "
                        "Separates 'the reference is already wrong' from 'the "
                        "optimizer walked away from a good reference'.")
    return p.parse_args()


def main() -> None:
    import mujoco

    args = parse_args()
    run = Path(args.run)
    side = args.side

    from retargeting.utils.mano_robot_map import (
        _bid, _canonical, _palmar_world, _surface_world, segment_frame, segments)

    model = mujoco.MjModel.from_xml_path(args.scene or str(run / "scene.xml"))
    data = mujoco.MjData(model)
    traj = np.load(run / "trajectory_mjwp.npz", allow_pickle=True)
    # qpos is saved per control chunk, (chunks, steps_per_chunk, nq).
    qpos = np.asarray(traj["qpos"], float).reshape(-1, model.nq)
    if args.reference:
        if "qpos_ref" not in traj.files:
            raise SystemExit(
                f"{run}/trajectory_mjwp.npz has no qpos_ref; it predates the "
                "change that saves it. Re-run the optimization.")
        qpos = np.asarray(traj["qpos_ref"], float).reshape(-1, model.nq)
        print("scoring the TRACKED REFERENCE, not the optimized trajectory")
    n_steps = qpos.shape[0]

    warm = (np.asarray(traj["warmup_progress"], float).reshape(-1)
            if "warmup_progress" in traj.files else None)
    start = args.from_step
    if start is None:
        start = (int(np.argmax(warm >= 1.0))
                 if warm is not None and len(warm) == n_steps and (warm >= 1.0).any()
                 else 0)
    print(f"trajectory {n_steps} steps; analysing from {start} (post-warmup)\n")

    # Palmar axis of each link, in that link's own frame so it stays valid as
    # the hand moves. Taken at the open pose, where the frames are defined.
    cd = _canonical(model, side)
    axis, centre = {}, {}
    for seg in segments(side):
        b = _bid(model, seg["rob_frame"])
        fr = segment_frame(model, cd, seg["rob_frame"], seg["rob_distal"],
                           _palmar_world(model, side, seg["finger"]),
                           _surface_world(model, cd, seg["rob_surf"]))
        R = cd.xmat[b].reshape(3, 3)
        for nm in seg["rob_surf"]:
            axis[_bid(model, nm)] = (R.T @ fr["z"], seg["name"], seg["finger"])
            centre[_bid(model, nm)] = fr["cz"]
        axis[b] = (R.T @ fr["z"], seg["name"], seg["finger"])
        centre[b] = fr["cz"]

    m = np.load(args.map, allow_pickle=True)
    map_body = m["body_id"].astype(np.int64)
    map_off = np.asarray(m["local_offset"], float)

    obj_bid = _bid(model, f"{side}_object")
    is_obj = np.asarray(model.geom_bodyid) == obj_bid

    n_frames = 0
    palmar_n = dorsal_n = 0
    per_link = Counter()
    palmar_per_finger = Counter()
    dorsal_per_finger = Counter()
    dists = []
    frames_with_contact = 0

    for s in range(start, n_steps, args.stride):
        data.qpos[:] = qpos[s]
        mujoco.mj_forward(model, data)
        n_frames += 1

        pts = data.xpos[map_body] + np.einsum(
            "nij,nj->ni", data.xmat[map_body].reshape(-1, 3, 3), map_off)

        touched = False
        for ci in range(data.ncon):
            c = data.contact[ci]
            if c.dist > 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if is_obj[g1] == is_obj[g2]:
                continue                       # not a hand<->object pair
            hand_geom = g2 if is_obj[g1] else g1
            b = int(model.geom_bodyid[hand_geom])
            if b not in axis:
                continue                       # arm base etc., not a hand link
            touched = True
            z_local, seg_name, finger = axis[b]
            rel = data.xmat[b].reshape(3, 3).T @ (c.pos - data.xpos[b])
            palmar = float(rel @ z_local) - centre[b]

            per_link[seg_name] += 1
            if palmar > 0:
                palmar_n += 1
                palmar_per_finger[finger] += 1
            else:
                dorsal_n += 1
                dorsal_per_finger[finger] += 1
            dists.append(float(np.linalg.norm(pts - c.pos, axis=1).min()))
        frames_with_contact += int(touched)

    total = palmar_n + dorsal_n
    print(f"frames sampled            : {n_frames}")
    print(f"frames with hand-object contact: {frames_with_contact} "
          f"({100 * frames_with_contact / max(n_frames, 1):.0f}%)")
    print(f"hand-object contacts      : {total}")
    if total == 0:
        print("\nNO CONTACT AT ALL — a zero repel term meant the hand never "
              "touched the object, not that it touched it in the right places.")
        return

    print(f"\n  PALMAR (pad side) : {palmar_n:5d}  {100 * palmar_n / total:5.1f}%")
    print(f"  DORSAL (back)     : {dorsal_n:5d}  {100 * dorsal_n / total:5.1f}%")

    print("\nby finger:")
    print("%-8s %8s %8s %8s" % ("finger", "palmar", "DORSAL", "dorsal%"))
    for f in ("thumb", "index", "middle", "ring", "pinky", "middle"):
        p, dd = palmar_per_finger[f], dorsal_per_finger[f]
        if p + dd == 0:
            continue
        print("%-8s %8d %8d %7.1f%%" % (f, p, dd, 100 * dd / (p + dd)))
        palmar_per_finger[f] = dorsal_per_finger[f] = 0   # avoid the dup entry

    print("\nby link:  " + "  ".join(f"{k}:{v}" for k, v in per_link.most_common()))

    d = np.asarray(dists)
    print(f"\ndistance from each contact to the nearest demonstrated point:")
    print(f"  mean {100 * d.mean():.2f} cm   median {100 * np.median(d):.2f} cm   "
          f"p90 {100 * np.percentile(d, 90):.2f} cm")
    print(f"  within the 2 cm free radius: {100 * (d < 0.02).mean():.0f}%")


if __name__ == "__main__":
    main()
