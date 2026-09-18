"""Fill NaN frames in a HaWoR hand track by holding the nearest valid frame.

HaWoR estimates articulation (``*_rot``, ``*_hand_pose``) for every frame but
can fail to place the hand in space, leaving ``*_trans``/``*_betas`` — and so
``*_joints``/``*_vertices`` — as NaN. Measured on metalcupmove: 31 of 301
frames, in runs at both ends (0-14, 23-27, 289-299), while rot and hand_pose
were fine throughout. The ``*_valid`` flags do not catch this: 300 of them were
True on frames whose vertices are entirely NaN.

Downstream this surfaces as ``numpy.linalg.LinAlgError: SVD did not converge``
from ``Rotation.from_matrix`` in process_dataset, which names neither the frame
nor the cause.

Trimming is the obvious fix and the wrong one here: the object layout JSON is
indexed by the same frame numbers, so dropping hand frames desynchronises the
two. This holds the nearest valid frame instead, keeping the frame count and
the alignment. The filled frames are wrong — the hand does not actually stand
still there — so this is only safe when the NaN runs sit outside the interaction
(metalcupmove grasps from frame 48, well inside the valid 28-288 run). The
report below prints the runs so that can be checked before trusting the output.

    python repair_hand_nan.py dataset/metalcupmove/metalcupmove/all_hand_meshes.npz
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end] index runs where ``mask`` is True."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            out.append((int(s), int(p)))
            s = i
        p = i
    out.append((int(s), int(p)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", help="all_hand_meshes.npz to repair (in place).")
    p.add_argument("--sides", default="left,right")
    p.add_argument("--min-valid", type=int, default=30,
                   help="A side with fewer valid frames than this is treated as "
                        "absent: its *_valid flags are cleared so the pipeline's "
                        "own guard drops it, and its arrays are left alone. "
                        "Filling an absent hand from one stray frame would "
                        "manufacture a stationary phantom hand instead.")
    p.add_argument("--report", action="store_true", help="Measure only; write nothing.")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    path = Path(args.npz)
    data = dict(np.load(path))
    n = None
    touched = False

    for side in args.sides.split(","):
        key = f"{side}_vertices"
        if key not in data:
            continue
        V = np.asarray(data[key], float)
        n = V.shape[0]
        good = np.isfinite(V).reshape(n, -1).all(1)
        if good.all():
            print(f"{side}: {n} frames, no NaN")
            continue
        if int(good.sum()) < args.min_valid:
            print(f"{side}: only {int(good.sum())}/{n} frames are finite — treating "
                  f"as absent (below --min-valid {args.min_valid})")
            if not args.report and f"{side}_valid" in data:
                data[f"{side}_valid"] = np.zeros(n, bool)
                print(f"  cleared {side}_valid so the pipeline drops this hand")
                touched = True
            continue
        print(f"{side}: {n} frames, {int((~good).sum())} NaN")
        print(f"  NaN runs   : {runs_of(~good)}")
        print(f"  valid runs : {runs_of(good)}")
        longest = max(runs_of(good), key=lambda r: r[1] - r[0])
        print(f"  longest valid run: {longest}  "
              f"({longest[1] - longest[0] + 1} frames)")
        if args.report:
            continue

        # The *_valid flags do not track this: on metalcupmove 300 of them were
        # True on frames whose vertices are entirely NaN. AND them with what the
        # data actually contains, so downstream guards see the truth.
        if f"{side}_valid" in data:
            v = np.asarray(data[f"{side}_valid"], bool)
            if (v & ~good).any():
                print(f"  {side}_valid claimed {int((v & ~good).sum())} NaN frames "
                      f"were valid; corrected")
                data[f"{side}_valid"] = v & good
                touched = True

        # Nearest valid frame for every index, i.e. edge-clamped hold.
        gi = np.flatnonzero(good)
        src = gi[np.abs(np.arange(n)[:, None] - gi[None, :]).argmin(1)]
        for k, a in list(data.items()):
            a = np.asarray(a)
            if not k.startswith(f"{side}_") or a.ndim < 1 or a.shape[0] != n:
                continue
            if not np.issubdtype(a.dtype, np.floating):
                continue
            bad = ~np.isfinite(a).reshape(n, -1).all(1)
            if bad.any():
                a = a.copy()
                a[bad] = a[src[bad]]
                data[k] = a
                print(f"  filled {k}: {int(bad.sum())} frames")
                touched = True

    if args.report or not touched:
        print("\nnothing written" if not args.report else "\n--report: nothing written")
        return
    if not args.no_backup:
        bak = path.with_suffix(path.suffix + ".orig")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"\nbacked up original to {bak}")
    np.savez(path, **data)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
