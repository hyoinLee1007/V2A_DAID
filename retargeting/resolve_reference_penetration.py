"""Push the retargeted reference out of the object, by moving joints.

The IK reference tracked in physics buries the robot hand in the cup: 11.0 mm
of penetration on average, 18.0 mm at worst, past the 5 mm margin in 75% of
steps. Physics cannot reproduce that, so the optimizer backs the hand off, and
backing off from a buried pad puts the *back* of the finger on the handle.
Measured across four runs spanning 125x in reward weights, the middle finger
was 100% dorsal every time — the reward was never the cause.

Only about 3.3 mm of that is the hand track: the MANO hand at the ray-corrected
pose penetrates 3.34 mm mean / 4.93 mm max, already inside the margin. The rest
is retargeting — the robot's links are thicker than the fingers they stand in
for, so putting its joints where MANO's were buries it deeper.

Rigid translation cannot fix it. Measured: clearing the margin needs 14.8 mm of
shift on average and 28 mm at worst, and 3 of 24 sampled frames are unsolvable
within 30 mm, because the penetration is spread over several links and pushing
one out drives another in. Joints have to move.

So this solves, per frame,

    min_q   sum over hand-object contacts of max(0, -dist - margin)^2
            + lambda * ||q_hand - q_ref||^2

over the hand DOFs only, warm-started from the previous frame so the corrected
trajectory stays smooth. The object DOFs are never touched: the cup's pose is
the demonstration's, not something to negotiate with.

    python resolve_reference_penetration.py \
        --run outputs/sharpa/left/cupmove_refsave/0 --report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/sharpa/left/cupmove_refsave/0")
    p.add_argument("--scene", default=None,
                   help="scene.xml to use (archived runs lose their asset paths).")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--key", default="qpos_ref",
                   help="Array in trajectory_mjwp.npz to correct.")
    p.add_argument("--margin", type=float, default=0.005,
                   help="Allowed penetration, matching penetration_margin.")
    p.add_argument("--stay", type=float, default=200.0,
                   help="lambda: weight on staying at the reference pose. Higher "
                        "keeps the demonstrated shape and clears less.")
    p.add_argument("--palmar", type=float, default=1.0,
                   help="mu: weight on ending up pad-side rather than back-side. "
                        "Same mm^2 scale as the penetration term.")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--step", type=float, default=0.02,
                   help="Gradient step, rad per unit cost.")
    p.add_argument("--fd", type=float, default=5e-3,
                   help="Finite-difference step, rad.")
    p.add_argument("--wrist", action="store_true",
                   help="Also move the wrist. Off by default: the wrist carries "
                        "the approach direction and moving it changes which "
                        "finger reaches the handle.")
    p.add_argument("--stride", type=int, default=1,
                   help="Correct every Nth step and interpolate between (the "
                        "reference is sampled at sim_dt, so neighbours are "
                        "nearly identical and solving each is wasted work).")
    p.add_argument("--report", action="store_true",
                   help="Only measure; do not write anything.")
    p.add_argument("--out", default=None,
                   help="npz to write the corrected array to "
                        "(default: <run>/reference_depenetrated.npz).")
    return p.parse_args()


def make_probe(model, data, side):
    """``probe(qpos, margin) -> (deepest, excess, dorsal)`` for hand<->object.

    ``dorsal`` is the summed squared depth of contacts made with the *back* of a
    link, in mm, on the same scale as ``excess``. Without it the correction
    pushes each link out along whichever direction is cheapest, and for a finger
    already wrapped past the handle that direction leaves the back against the
    object: measured on the first version, the reference's dorsal share went
    from 23% to 47% on the middle finger and 16% to 68% on the ring, and the
    optimized grasp moved onto the pinky's knuckles.
    """
    import mujoco
    from retargeting.utils.mano_robot_map import (
        _bid, _canonical, _palmar_world, _surface_world, segment_frame, segments)

    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    is_hand = np.array([b != ob and b != 0 for b in model.geom_bodyid])
    is_obj = model.geom_bodyid == ob

    # Palmar axis of each link in its own frame, plus the mid-surface offset
    # along it, read at the open pose — the same construction the reward's
    # repel term uses, so "dorsal" means the same thing in both places.
    cd = _canonical(model, side)
    axis = np.zeros((model.nbody, 3))
    ctr = np.zeros(model.nbody)
    known = np.zeros(model.nbody, bool)
    for seg in segments(side):
        fb = _bid(model, seg["rob_frame"])
        fr = segment_frame(model, cd, seg["rob_frame"], seg["rob_distal"],
                           _palmar_world(model, side, seg["finger"]),
                           _surface_world(model, cd, seg["rob_surf"]))
        z_local = cd.xmat[fb].reshape(3, 3).T @ fr["z"]
        for nm in list(seg["rob_surf"]) + [seg["rob_frame"]]:
            b = _bid(model, nm)
            axis[b], ctr[b], known[b] = z_local, fr["cz"], True

    def probe(qpos, margin):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        deepest, excess, dorsal = 0.0, 0.0, 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = con.geom1, con.geom2
            if not ((is_hand[g1] and is_obj[g2]) or (is_obj[g1] and is_hand[g2])):
                continue
            if con.dist >= 0.0:
                continue
            b = int(model.geom_bodyid[g1] if is_hand[g1] else model.geom_bodyid[g2])
            if known[b]:
                rel = con.pos - data.xpos[b]
                pal = (data.xmat[b].reshape(3, 3).T @ rel) @ axis[b] - ctr[b]
                if pal < 0.0:
                    dorsal += (1000.0 * -pal) ** 2
            pen = -con.dist
            deepest = max(deepest, pen)
            over = pen - margin
            if over > 0.0:
                # In MILLIMETRES. In metres this term is ~1e-4 while the
                # stay-at-the-reference term is ~1e-3 rad^2, so the solver
                # correctly concluded that not moving was cheaper and returned
                # the input unchanged (measured: 0.02 deg of motion, 17.77 mm
                # of penetration left). mm^2 puts the two on one scale, and
                # lambda then reads as "mm^2 of penetration per rad^2 of pose
                # change I am willing to trade".
                excess += (1000.0 * over) ** 2
        return deepest, excess, dorsal

    return probe


def main() -> None:
    import mujoco

    args = parse_args()
    run = Path(args.run)
    model = mujoco.MjModel.from_xml_path(args.scene or str(run / "scene.xml"))
    data = mujoco.MjData(model)
    probe = make_probe(model, data, args.side)

    traj = np.load(run / "trajectory_mjwp.npz", allow_pickle=True)
    if args.key not in traj.files:
        raise SystemExit(f"{run}/trajectory_mjwp.npz has no '{args.key}'")
    Q = np.asarray(traj[args.key], float).reshape(-1, model.nq).copy()

    # Hand DOFs only. The object's free joint is the last 7 of qpos and stays
    # exactly as the demonstration had it.
    lo = 0 if args.wrist else 6
    dofs = np.arange(lo, model.nu)
    print(f"{len(Q)} steps, correcting {len(dofs)} DOFs "
          f"({'wrist + fingers' if args.wrist else 'fingers only'}), "
          f"margin {args.margin * 1000:.1f} mm, lambda {args.stay}, mu {args.palmar}")

    idx = list(range(0, len(Q), args.stride))
    before = np.zeros(len(idx))
    after = np.zeros(len(idx))
    moved = np.zeros(len(idx))
    Qc = Q.copy()
    warm = None

    for n, s in enumerate(idx):
        q0 = Q[s].copy()
        d0, _, _ = probe(q0, args.margin)
        before[n] = d0
        if d0 <= args.margin:
            after[n] = d0
            warm = None
            continue

        q = q0.copy()
        if warm is not None:
            # Start from the previous frame's correction: consecutive reference
            # steps differ by well under a degree, so the same joint change is
            # very nearly right and the descent starts most of the way there.
            q[dofs] = q0[dofs] + warm

        def cost(qv):
            _, exc, dor = probe(qv, args.margin)
            dq = qv[dofs] - q0[dofs]
            return exc + args.palmar * dor + args.stay * float(dq @ dq)

        c = cost(q)
        for _ in range(args.iters):
            g = np.zeros(len(dofs))
            for k, j in enumerate(dofs):
                qp = q.copy()
                qp[j] += args.fd
                g[k] = (cost(qp) - c) / args.fd
            gn = np.linalg.norm(g)
            if gn < 1e-9:
                break
            trial = q.copy()
            trial[dofs] -= args.step * g / gn
            np.clip(trial[dofs], model.jnt_range[dofs, 0], model.jnt_range[dofs, 1],
                    out=trial[dofs])
            ct = cost(trial)
            if ct >= c:
                args.step *= 0.5          # backtrack; restored below
                if args.step < 1e-4:
                    args.step = 0.02
                    break
                continue
            q, c = trial, ct

        args.step = 0.02
        d1, _, _ = probe(q, args.margin)
        after[n] = d1
        moved[n] = float(np.linalg.norm(q[dofs] - q0[dofs]))
        warm = q[dofs] - q0[dofs]
        Qc[s] = q

    if args.stride > 1:
        # Only the DOFs the solver was allowed to move. Blending the whole row
        # rewrites DOFs this tool never touched, and the reference has a
        # single-step 191.4 mm wrist jump at the warmup boundary that a stride-10
        # blend silently smeared into a 10-step ramp — a real discontinuity
        # hidden by a tool that was supposed to leave the wrist alone.
        for a, b in zip(idx[:-1], idx[1:]):
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                Qc[k, dofs] = (1 - t) * Qc[a, dofs] + t * Qc[b, dofs]

    bad = before > args.margin
    print(f"\nsteps over the margin: {int(bad.sum())} of {len(idx)}")
    print(f"deepest penetration  before: mean {before.mean() * 1000:6.2f} mm  "
          f"max {before.max() * 1000:6.2f} mm")
    print(f"                     after : mean {after.mean() * 1000:6.2f} mm  "
          f"max {after.max() * 1000:6.2f} mm")
    print(f"still over the margin after: {int((after > args.margin).sum())} of {len(idx)}")
    if bad.any():
        print(f"joint change on corrected steps: mean {np.degrees(moved[bad].mean()):.2f} deg "
              f"(L2 over {len(dofs)} DOFs)  max {np.degrees(moved.max()):.2f} deg")

    if args.report:
        print("\n--report: nothing written")
        return
    out = Path(args.out) if args.out else run / "reference_depenetrated.npz"
    np.savez(out, qpos_ref=Qc, margin=args.margin, stay=args.stay,
             wrist=bool(args.wrist), source=str(run / "trajectory_mjwp.npz"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
