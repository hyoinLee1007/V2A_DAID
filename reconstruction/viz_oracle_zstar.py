"""Plot the oracle ray-depth correction produced by ``oracle_ray_depth.py``.

Four stacked panels over the frame axis:

  1. z*        the per-frame ray-depth correction. A consistent sign means a
               systematic depth bias (the signature of monocular depth
               ambiguity) rather than random per-frame noise.
  2. gap       min fingertip-to-object-surface distance, before vs after.
               This is the go/no-go quantity: if "after" does not sit near
               zero on the contact frames, a 1-D ray correction is not enough.
  3. anchors   CHOIR-style accepted correspondences (within 2 cm, hand normal
               inside a 60 deg cone), before vs after. This is the measure that
               does not presume which fingers touch.
  4. status    which frames were used. Frames are excluded when the raw hand is
               farther than ``interaction_thresh`` (approach / retreat / tracking
               dropout) or when the 1-D search saturated at the scan boundary.

Usage:
    python viz_oracle_zstar.py --npz heatmap_out/oracle_zstar_cupmove.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default="heatmap_out/oracle_zstar_cupmove.npz")
    p.add_argument("--out", default=None, help="Output PNG (default: <npz>.png).")
    p.add_argument("--gate", type=float, default=0.5,
                   help="Gap threshold (cm) drawn as the pass line on panel 2.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.npz)
    f = d["frames"]
    good = d["good"] if "good" in d.files else np.ones(len(f), bool)
    interacting = d["interacting"] if "interacting" in d.files else good
    saturated = d["saturated"] if "saturated" in d.files else np.zeros(len(f), bool)

    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 2.5, 0.9]})

    def shade(a):
        """Grey out every frame that was excluded from the summary."""
        bad = ~good
        if bad.any():
            start = None
            for i, b in enumerate(list(bad) + [False]):
                if b and start is None:
                    start = i
                elif not b and start is not None:
                    a.axvspan(f[start] - 0.5, f[i - 1] + 0.5, color="0.85", zorder=0)
                    start = None

    # --- 1. z* ---
    a = ax[0]; shade(a)
    z = d["z_star"] * 100
    a.plot(f, z, lw=1.6, color="tab:blue", marker="o", ms=3, label="z* (all frames)")
    a.plot(f[good], z[good], lw=0, marker="o", ms=4, color="tab:red",
           label="z* (used)")
    a.axhline(0, color="k", lw=0.8)
    if good.any():
        m = float(np.nanmean(z[good]))
        a.axhline(m, color="tab:red", ls="--", lw=1,
                  label="mean over used = %+.2f cm" % m)
    a.set_ylabel("z*  (cm along camera ray)")
    a.set_title("Oracle ray-depth correction  —  %s" % Path(args.npz).name)
    a.grid(alpha=0.3); a.legend(fontsize=8)

    # --- 2. gap before/after ---
    a = ax[1]; shade(a)
    a.plot(f, d["gap_before"], lw=1.5, color="tab:orange", marker="o", ms=3,
           label="before")
    a.plot(f, d["gap_after"], lw=1.5, color="tab:green", marker="o", ms=3,
           label="after")
    a.axhline(args.gate, color="tab:red", ls="--", lw=1,
              label="gate = %.1f cm" % args.gate)
    a.set_yscale("symlog", linthresh=0.1)
    a.set_ylabel("min fingertip gap (cm)")
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)

    # --- 3. anchors ---
    a = ax[2]; shade(a)
    if "anchors_after" in d.files and np.any(d["anchors_after"] > 0):
        a.bar(f - 0.2, d["anchors_before"], width=0.4, color="tab:orange", label="before")
        a.bar(f + 0.2, d["anchors_after"], width=0.4, color="tab:green", label="after")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "run oracle_ray_depth.py with --report-anchors",
               ha="center", va="center", transform=a.transAxes, color="0.5")
    a.set_ylabel("accepted anchors\n(<2 cm, <60 deg)")
    a.grid(alpha=0.3)

    # --- 4. status strip ---
    a = ax[3]
    status = np.zeros(len(f))
    status[interacting] = 1.0
    status[saturated] = 0.5
    a.imshow(status[None, :], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
             extent=[f[0] - 0.5, f[-1] + 0.5, 0, 1])
    a.set_yticks([])
    a.set_xlabel("frame")
    a.set_title("green = interaction frame used   |   red = excluded "
                "(hand too far, or 1-D search saturated)", fontsize=9)

    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".png")
    fig.savefig(str(out), dpi=130)
    print(f"Saved {out}")

    if good.any():
        gb, ga = d["gap_before"][good], d["gap_after"][good]
        print("used frames: %d / %d" % (int(good.sum()), len(f)))
        print("  gap  before mean %.2f cm  ->  after mean %.2f cm" % (gb.mean(), ga.mean()))
        print("  z*   mean %+.2f cm   std %.2f cm" % (z[good].mean(), z[good].std()))
        if "anchors_after" in d.files:
            print("  anchors  before mean %.1f  ->  after mean %.1f"
                  % (d["anchors_before"][good].mean(), d["anchors_after"][good].mean()))


if __name__ == "__main__":
    main()
