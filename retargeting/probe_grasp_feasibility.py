"""Can this robot hand hold the mug the way the demonstration does — at all?

The trajectory optimizer never reaches the demonstrated grasp: across nine
things we changed and measured (penetration, reward scales over a 125x sweep,
reference dorsal quality, the warmup handoff, finger thickness, joint limits,
the entry path, the palmar candidate filter, and per-finger weighting) the
middle finger came out 100% back-of-finger every time. Every one of those was a
reason it *might* have failed on the way in. None of them was the reason.

So ask the question without the trajectory. Start from the demonstrated pose
itself — the retargeted reference, which penetrates the cup but has both
fingers already inside the handle loop — and optimise the pose alone until it
is physically valid. Where it converges is the answer:

  it converges to the demonstrated grasp   the hand can do it, and the failure
                                           is the trajectory optimizer's search
  it converges elsewhere                   the demonstrated grasp is not a
                                           valid pose for this hand, and
                                           reproducing it is the wrong target

Starting from the penetrating pose is the point. That pose encodes "both
fingers are through the handle", which is exactly what an optimizer approaching
from outside cannot discover, and from there the work is only to back out.

    min_q  sum max(0, penetration - margin)^2          be physically valid
         + mu * sum max(0, -palmar)^2                  touch with the pad
         + nu * weighted_mean_k min_v ||p_k - x_v||    touch where the demo did

The third term is what makes this a test of *the* grasp rather than of any
grasp: without it the cheapest answer is an open hand held away from the cup,
which is valid, converged, and meaningless. Its targets are the anchor pairs —
each mapped skin point carries the object vertices it actually met, so "did
point k reach its own target" is answerable per point after convergence.

This can show the grasp is possible (by exhibiting a pose). It cannot show it
is impossible: a failed local search and a genuine impossibility look the same,
which is what --restarts is for, and why a negative result here should be
confirmed with a real grasp synthesiser rather than trusted.

    python probe_grasp_feasibility.py --step 900 --restarts 20
    python probe_grasp_feasibility.py --step 900 --wrist-free --hold-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/sharpa/left/cupmove_decay2/0")
    p.add_argument("--scene", default=None)
    p.add_argument("--map", default="outputs/robot_contact_map_left.npz")
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--key", default="qpos_ref_raw",
                   help="Array to take the initial pose from. The default is the "
                        "reference before any of our corrections — the one that "
                        "still penetrates, and so still has both fingers inside "
                        "the loop.")
    p.add_argument("--step", type=int, default=900, help="Which step to start from.")
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("--pen-weight", type=float, default=1.0,
                   help="Weight on the penetration term. At 1.0 it is swamped: "
                        "7.5 mm of penetration scores 6 against 214 for the "
                        "target term, so the solver leaves the pose penetrating "
                        "and the hold test then launches the cup on contact "
                        "impulse rather than testing the grasp.")
    p.add_argument("--mu", type=float, default=1.0, help="Dorsal-contact weight.")
    p.add_argument("--nu", type=float, default=20.0,
                   help="Weight on reaching the demonstrated targets, in cost per "
                        "mm of weighted-mean point-to-target distance.")
    p.add_argument("--wrist-free", action="store_true",
                   help="Also optimise the 6 wrist DOFs. Fixed asks 'can it be "
                        "done from the human's wrist placement', free asks 'can "
                        "this hand do it'; the gap between the two is the answer "
                        "to whether retargeting placed the wrist badly.")
    p.add_argument("--restarts", type=int, default=1,
                   help="Extra runs from the initial pose perturbed by --jitter. "
                        "One local search failing proves nothing.")
    p.add_argument("--jitter", type=float, default=0.15, help="Restart noise, rad.")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--fd", type=float, default=5e-3)
    p.add_argument("--hold-test", action="store_true",
                   help="After converging, drop the weld and run gravity for "
                        "--hold-seconds; report how far the cup moves. A pose can "
                        "be valid and still not hold.")
    p.add_argument("--hold-seconds", type=float, default=1.5)
    p.add_argument("--save", default=None,
                   help="Write the converged qpos to this npz so it can be looked "
                        "at. Without it the only evidence a pose exists is the "
                        "objective's own numbers, which is no evidence at all.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    import mujoco
    from resolve_reference_penetration import make_probe
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    from retargeting.utils.paired_contact_cost import build_targets
    from retargeting.utils.mano_robot_map import _bid

    args = parse_args()
    run = Path(args.run)
    model = mujoco.MjModel.from_xml_path(args.scene or str(run / "scene.xml"))
    data = mujoco.MjData(model)
    probe = make_probe(model, data, args.side)
    obj = _bid(model, f"{args.side}_object")

    traj = np.load(run / "trajectory_mjwp.npz", allow_pickle=True)
    if args.key not in traj.files:
        raise SystemExit(f"{args.key} not in {run}/trajectory_mjwp.npz")
    q_init = np.asarray(traj[args.key], float).reshape(-1, model.nq)[args.step].copy()

    # Mapped skin points and, for each, the object vertices it actually met.
    cm = np.load(args.map, allow_pickle=True)
    body = cm["body_id"].astype(np.int64)
    offset = np.asarray(cm["local_offset"], float)
    weight = np.asarray(cm["weight"], float)
    finger = cm["finger"].astype(str)
    cap = 64
    keep = np.argpartition(-weight, cap)[:cap] if len(weight) > cap else np.arange(len(weight))
    body, offset, weight, finger = body[keep], offset[keep], weight[keep], finger[keep]
    V = _visual_mesh_local_verts(model, args.side)
    built = build_targets(cm["tgt_offset"], cm["tgt_vertex"], cm["tgt_weight"],
                          keep, V, 0.0, "cpu", torch.float64)
    if built is None:
        raise SystemExit(f"{args.map} has no target lists; rebuild it")
    tgt_xyz, tgt_off, tgt_empty = built
    wt = torch.tensor(weight)
    live = ~tgt_empty

    def target_dist(qpos):
        """(weighted mean, per-point) distance to each point's own targets, metres."""
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        R = data.xmat[body].reshape(-1, 3, 3)
        P = data.xpos[body] + np.einsum("nij,nj->ni", R, offset)
        loc = torch.tensor((P - data.xpos[obj]) @ data.xmat[obj].reshape(3, 3))[None]
        pp = (loc * loc).sum(-1, keepdim=True)
        xx = (tgt_xyz * tgt_xyz).sum(-1)[None]
        px = torch.einsum("wkd,ktd->wkt", loc, tgt_xyz)
        d = ((pp + xx - 2 * px).clamp_min(0).sqrt() + tgt_off[None]).amin(-1)[0]
        w = torch.where(live, wt, torch.zeros_like(wt))
        return float((d * w).sum() / w.sum().clamp_min(1e-12)), d.numpy()

    dofs = np.arange(0 if args.wrist_free else 6, model.nu)

    def cost(qv):
        _, exc, dor = probe(qv, args.margin)
        md, _ = target_dist(qv)
        return (args.pen_weight * exc + args.mu * dor
                + args.nu * (1000.0 * md)), exc, dor, md

    def descend(q):
        c = cost(q)[0]
        step = args.step_size
        for _ in range(args.iters):
            g = np.zeros(len(dofs))
            for k, j in enumerate(dofs):
                qp = q.copy(); qp[j] += args.fd
                g[k] = (cost(qp)[0] - c) / args.fd
            gn = np.linalg.norm(g)
            if gn < 1e-9:
                break
            trial = q.copy()
            trial[dofs] -= step * g / gn
            np.clip(trial[dofs], model.jnt_range[dofs, 0], model.jnt_range[dofs, 1],
                    out=trial[dofs])
            ct = cost(trial)[0]
            if ct >= c:
                step *= 0.5
                if step < 1e-4:
                    break
                continue
            q, c = trial, ct
        return q, c

    def report(tag, q):
        deep, exc, dor = probe(q, args.margin)
        md, per = target_dist(q)
        # palmar/dorsal split of the actual contacts, per finger
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        print(f"\n{tag}")
        print(f"  penetration deepest {deep * 1000:6.2f} mm   dorsal term {dor:9.1f}")
        print(f"  target distance, weighted mean {md * 100:6.2f} cm")
        print(f"  {'finger':8s} {'pts':>4s} {'dist to own target':>19s} {'<1cm':>6s}")
        for f in sorted(set(finger)):
            k = finger == f
            print(f"  {f:8s} {int(k.sum()):4d} {per[k].mean() * 100:16.2f} cm "
                  f"{int((per[k] < 0.01).sum()):6d}")

    rng = np.random.default_rng(args.seed)
    print(f"start: {args.key}[{args.step}], "
          f"{'wrist + fingers' if args.wrist_free else 'fingers only'} "
          f"({len(dofs)} DOF), mu={args.mu} nu={args.nu}")
    report("INITIAL (demonstrated pose)", q_init)

    best, best_c = None, np.inf
    for r in range(max(1, args.restarts)):
        q = q_init.copy()
        if r > 0:
            q[dofs] += rng.normal(scale=args.jitter, size=len(dofs))
            np.clip(q[dofs], model.jnt_range[dofs, 0], model.jnt_range[dofs, 1],
                    out=q[dofs])
        q, c = descend(q)
        if c < best_c:
            best, best_c = q, c
        print(f"  restart {r:2d}: cost {c:12.1f}"
              + ("   <- best" if c == best_c else ""))

    report(f"CONVERGED (best of {max(1, args.restarts)})", best)

    if args.save:
        deep, exc, dor = probe(best, args.margin)
        md, per = target_dist(best)
        np.savez(args.save, qpos=best, qpos_init=q_init, step=args.step,
                 penetration=deep, dorsal=dor, target_mean=md, target_per_point=per,
                 finger=finger, wrist_free=bool(args.wrist_free))
        print(f"\nsaved converged pose to {args.save}")

    if args.hold_test:
        data.qpos[:] = best
        data.qvel[:] = 0.0
        data.ctrl[:] = best[:model.nu]
        mujoco.mj_forward(model, data)
        p0 = data.xpos[obj].copy()
        R0 = data.xmat[obj].reshape(3, 3).copy()
        for _ in range(int(args.hold_seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        drop = data.xpos[obj] - p0
        tilt = np.degrees(np.arccos(np.clip(
            (np.trace(R0.T @ data.xmat[obj].reshape(3, 3)) - 1) / 2, -1, 1)))
        print(f"\nHOLD TEST ({args.hold_seconds:.1f}s under gravity)")
        print(f"  cup moved {np.linalg.norm(drop) * 1000:6.1f} mm "
              f"(z {drop[2] * 1000:+6.1f} mm), rotated {tilt:5.1f} deg")
        print("  -> " + ("held" if np.linalg.norm(drop) < 0.02 else "DROPPED"))


if __name__ == "__main__":
    main()
