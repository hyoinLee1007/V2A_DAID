"""Plot the per-term reward magnitudes recorded in ``trajectory_mjwp.npz``.

Answers "which objective term is actually driving this run?" — the question
that is impossible to eyeball from the scale coefficients alone, because the
terms have different functional forms (``qpos_rew`` is a weighted L2 *norm*,
the heatmap term is ``scale * sum of squared distances``, and so on).

What the numbers are
--------------------
``optimize()`` stores one row per control chunk and one column per MPPI
iteration; each entry is that term **summed over the rollout horizon** and
averaged over samples (``cum_info`` in sampling.py). So a value is "how much
this term contributed to a whole 3 s plan", not a per-step value — the same
convention for every term, which is what makes them comparable here.

Sign convention: reward terms (``qpos_rew``, ``qvel_rew``) are stored negative,
penalties positive. This plots |value| so bar heights are comparable, and the
stacked panel shows each term's share of the total absolute contribution.

Usage (from retargeting/, in the `retargeting` conda env):
    python viz_reward_terms.py --run-dir outputs/sharpa/left/cupmove/0
    python viz_reward_terms.py \
        --run-dir outputs/sharpa/left/cupmove/0 \
        --run-dir outputs/sharpa/left/cupmove_palmside_scale1/0 \
        --label "palm 1000" --label "palm 1"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Terms that make up the reward, in a stable plotting order. Anything present
# in the npz but not listed here is appended after these.
_PREFERRED = [
    "qpos_rew",
    "qvel_rew",
    "pen_penalty",
    "drop_penalty",
    "wrist_floor_penalty",
    "contact_region_penalty",
]
# Recorded for diagnostics but not summed into `reward` — excluded by default
# so the share panel stays a true decomposition.
_DIAGNOSTIC = {
    "qpos_dist", "qvel_dist", "rew",
    "base_pos", "base_rot", "joint", "obj_pos", "obj_rot",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", required=True,
                   help="Run directory holding trajectory_mjwp.npz (repeatable).")
    p.add_argument("--label", action="append", default=None,
                   help="Legend label per --run-dir (defaults to the dir name).")
    p.add_argument("--reduce", choices=("last", "mean", "first"), default="last",
                   help="How to collapse the MPPI-iteration axis (default: last "
                        "= the converged value for that chunk).")
    p.add_argument("--include-diagnostics", action="store_true",
                   help="Also plot terms that are logged but not part of `reward`.")
    p.add_argument("--out", default=None,
                   help="Output PNG (default: <first run-dir>/reward_terms.png).")
    return p.parse_args()


def _valid_iters(npz, nchunk: int, niter: int) -> np.ndarray:
    """Per-chunk count of MPPI iterations that actually ran.

    `optimize()` stops early once improvement falls below
    ``improvement_threshold``, leaving the remaining columns of the
    ``max_num_iterations``-wide arrays zero-filled. Taking column -1 would
    therefore report 0 for every term, so the real last iteration is recovered
    from ``opt_steps`` when present and otherwise from the last column where
    any term is nonzero.
    """
    if "opt_steps" in npz.files:
        steps = np.asarray(npz["opt_steps"]).reshape(-1)
        if steps.shape[0] >= nchunk:
            return np.clip(steps[:nchunk].astype(int), 1, niter)

    stacked = []
    for key in npz.files:
        if not key.endswith("_mean"):
            continue
        arr = np.asarray(npz[key], dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (nchunk, niter):
            stacked.append(np.abs(arr))
    if not stacked:
        return np.full(nchunk, niter, dtype=int)
    any_nonzero = np.max(np.stack(stacked, 0), axis=0) > 0  # (nchunk, niter)
    counts = np.where(
        any_nonzero.any(axis=1), any_nonzero.shape[1] - np.argmax(any_nonzero[:, ::-1], axis=1), 1
    )
    return np.clip(counts.astype(int), 1, niter)


def load_terms(run_dir: Path, reduce: str, include_diag: bool) -> dict[str, np.ndarray]:
    npz = np.load(str(run_dir / "trajectory_mjwp.npz"))
    shapes = [
        np.asarray(npz[k]).shape
        for k in npz.files
        if k.endswith("_mean") and np.asarray(npz[k]).ndim == 2
    ]
    if not shapes:
        return {}
    nchunk, niter = shapes[0]
    nvalid = _valid_iters(npz, nchunk, niter)

    out: dict[str, np.ndarray] = {}
    for key in npz.files:
        if not key.endswith("_mean"):
            continue
        name = key[: -len("_mean")]
        if not include_diag and name in _DIAGNOSTIC:
            continue
        arr = np.asarray(npz[key], dtype=np.float64)
        if arr.ndim != 2 or arr.shape != (nchunk, niter):
            continue
        if reduce == "last":
            vals = arr[np.arange(nchunk), nvalid - 1]
        elif reduce == "first":
            vals = arr[:, 0]
        else:
            vals = np.array([arr[i, : nvalid[i]].mean() for i in range(nchunk)])
        out[name] = vals
    return out


def order_terms(names: list[str]) -> list[str]:
    known = [n for n in _PREFERRED if n in names]
    rest = sorted(n for n in names if n not in _PREFERRED)
    return known + rest


def main() -> None:
    args = parse_args()
    run_dirs = [Path(r) for r in args.run_dir]
    labels = args.label or []
    if len(labels) < len(run_dirs):
        labels += [d.parent.name or d.name for d in run_dirs[len(labels):]]

    runs = [load_terms(d, args.reduce, args.include_diagnostics) for d in run_dirs]
    names = order_terms(sorted({n for r in runs for n in r}))
    if not names:
        raise SystemExit("No *_mean reward terms found in the npz.")

    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    nrun = len(runs)
    fig, axes = plt.subplots(
        3, nrun, figsize=(7.0 * nrun, 11.0), squeeze=False,
    )

    for ci, (terms, label) in enumerate(zip(runs, labels)):
        nchunk = max((len(v) for v in terms.values()), default=0)
        x = np.arange(1, nchunk + 1)

        # --- panel 1: absolute magnitude, symlog (terms span many decades) ---
        ax = axes[0][ci]
        for n in names:
            if n not in terms:
                continue
            v = np.abs(terms[n])
            if not np.any(v > 0):
                continue  # a disabled term would just sit on the axis
            ax.plot(x, v, marker="o", ms=3, lw=1.6, label=n, color=colors[n])
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title(f"{label}\n|term| per chunk  ({args.reduce} MPPI iteration)")
        ax.set_xlabel("control chunk")
        ax.set_ylabel("|value|  (summed over horizon)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)

        # --- panel 2: share of total absolute contribution ---
        ax = axes[1][ci]
        stack_names = [n for n in names if n in terms and np.any(np.abs(terms[n]) > 0)]
        if stack_names:
            mat = np.stack([np.abs(terms[n]) for n in stack_names], axis=0)
            total = mat.sum(axis=0)
            total[total == 0] = 1.0
            share = 100.0 * mat / total
            ax.stackplot(x, share, labels=stack_names,
                         colors=[colors[n] for n in stack_names])
            ax.set_ylim(0, 100)
        ax.set_title("share of total |contribution|  (%)")
        ax.set_xlabel("control chunk")
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper right")

        # --- panel 3: dominance ranking, final chunk ---
        ax = axes[2][ci]
        finals = [(n, float(np.abs(terms[n][-1]))) for n in names if n in terms]
        finals.sort(key=lambda kv: kv[1], reverse=True)
        bars = [f for f in finals if f[1] > 0]
        if bars:
            ax.barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1],
                    color=[colors[b[0]] for b in bars][::-1])
            ax.set_xscale("log")
        ax.set_title("final chunk, largest term first")
        ax.set_xlabel("|value| (log)")
        ax.grid(alpha=0.3, axis="x", which="both")

    fig.tight_layout()
    out = Path(args.out) if args.out else run_dirs[0] / "reward_terms.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130)
    print(f"Saved {out}")

    # Text table so the numbers are quotable without opening the image.
    for terms, label, d in zip(runs, labels, run_dirs):
        print(f"\n=== {label}  ({d}) ===")
        rows = [(n, terms[n]) for n in names if n in terms]
        rows.sort(key=lambda kv: float(np.abs(kv[1][-1])), reverse=True)
        for n, v in rows:
            head = " ".join(f"{x:9.3f}" for x in v[: min(6, len(v))])
            print(f"  {n:26s} final={v[-1]:12.4f}   first chunks: {head}")


if __name__ == "__main__":
    main()
