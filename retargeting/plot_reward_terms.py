#!/usr/bin/env python3
"""최적화 중 보상 항의 크기를 청크(시간)별로 그린다. 논문 그림 스타일.

각 청크에서 MPPI 가 마지막 반복에 샘플한 궤적들의 **중앙값**을 쓴다. 평균은
4096 개 샘플 중 튄 몇 개에 끌려가서(예: 접촉항 평균 14.07 vs 중앙값 0.15)
항의 전형적인 크기를 보여주지 못한다. 보상은 음수이므로 비용(양수)으로 그린다.

    python plot_reward_terms.py \
        --run outputs/sharpa/right/metalcupmove_lam0/0 \
        --lam 0 --out outputs/figures/reward_metalcupmove_lam0

    # 두 조건을 같은 y 축으로
    python plot_reward_terms.py \
        --run outputs/sharpa/right/metalcupmove_lam0/0 \
              outputs/sharpa/right/metalcupmove_manual_lam5.0/0 \
        --lam 0 5 --out outputs/figures/reward_metalcupmove
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 첨부 그림(LNCS 스타일): 세리프, 검정 점선 + 빨강 실선, 테두리 없는 범례,
# 위·오른쪽 축선 없음.
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "savefig.dpi": 300,
}

# (정보 키, 부호, 범례, 선 스타일)
TERMS = [
    ("qpos_rew", -1.0, r"Tracking cost $\|\mathbf{w}\odot\Delta\mathbf{q}\|_2$",
     dict(color="black", ls="--", lw=1.1)),
    ("robot_contact_attract", 1.0, r"Contact cost $\lambda_c L_{\mathrm{contact}}$",
     dict(color="#C0504D", ls="-", lw=1.3)),
    ("pedestal_{side}", 1.0, r"Pedestal penalty $\lambda_{\mathrm{ped}} P_{\mathrm{ped}}$",
     dict(color="#7F7F7F", ls=":", lw=1.2)),
]


def load_terms(run: Path) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    z = np.load(run / "trajectory_mjwp.npz")
    n = z["opt_steps"].ravel().astype(int)
    last = lambda k: np.array([z[k][c, n[c] - 1] for c in range(len(n))])  # noqa: E731
    side = "right" if "pedestal_right_median" in z.files else "left"
    import yaml
    sim_dt = float(yaml.safe_load(open(run / "config.yaml"))["sim_dt"])
    t = z["sim_step"].ravel() * sim_dt                       # 청크 끝 시각 (s)
    out = {}
    for key, sign, *_ in TERMS:
        k = key.format(side=side) + "_median"
        out[key] = sign * last(k) if k in z.files else np.zeros(len(n))
    wp = z["warmup_progress"].ravel()
    warm_end = float(t[np.argmax(wp >= 1.0)] - (t[1] - t[0])) if (wp >= 1.0).any() else 0.0
    return t, out, warm_end


def draw(ax, t, terms, warm_end, lam, ymax):
    ax.axvspan(0, warm_end, color="0.92", lw=0, zorder=0)
    ax.text(warm_end / 2, ymax * 0.97, "warmup", ha="center", va="top",
            fontsize=8, color="0.45")
    for n_term, (key, _, label, style) in enumerate(TERMS):
        y = terms[key]
        if key == "robot_contact_attract" and lam == 0:
            continue                                           # 베이스라인엔 없는 항
        ax.plot(t, np.minimum(y, ymax), label=label, **style)
        # y 축을 넘는 값은 잘라서 그리고, 잘린 구간의 봉우리 값만 적는다.
        # 항마다 좌/우로 나눠 글자가 겹치지 않게 한다.
        right = n_term % 2 == 1
        for i, (ti, yi) in enumerate(zip(t, y)):
            prev_ = y[i - 1] if i > 0 else -np.inf
            next_ = y[i + 1] if i + 1 < len(y) else -np.inf
            if yi > ymax and yi >= prev_ and yi >= next_:
                ax.annotate(f"{yi:.2f}" if yi < 10 else f"{yi:.1f}", (ti, ymax),
                            xytext=(4 if right else -4, -2 - 9 * n_term),
                            textcoords="offset points", fontsize=7,
                            color=style["color"], va="top",
                            ha="left" if right else "right")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cost")
    ax.legend(loc="upper right", handlelength=2.6, fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--lam", nargs="+", type=float, required=True,
                   help="각 --run 의 lambda_c (범례·제목용)")
    p.add_argument("--ymax", type=float, default=None,
                   help="y 축 상한. 기본: 스파이크를 뺀 값의 95%% 분위수 기준")
    p.add_argument("--out", required=True, help="확장자 없이. .png 와 .pdf 저장")
    a = p.parse_args()
    if len(a.run) != len(a.lam):
        raise SystemExit("--run 과 --lam 개수가 다릅니다")

    data = [load_terms(Path(r)) for r in a.run]
    if a.ymax is None:
        vals = np.concatenate([v for _, terms, _ in data for v in terms.values()])
        a.ymax = float(np.ceil(np.percentile(vals, 95) * 1.6 * 20) / 20)

    plt.rcParams.update(STYLE)
    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(5.6, 2.3 * n), squeeze=False)
    for i, ((t, terms, warm_end), lam) in enumerate(zip(data, a.lam)):
        ax = axes[i, 0]
        draw(ax, t, terms, warm_end, lam, a.ymax)
        if n > 1:
            name = "DAID" if lam == 0 else rf"DAID + contact map ($\lambda_c={lam:g}$)"
            ax.set_title(f"({chr(97 + i)}) {name}", fontsize=9, loc="left")
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    print(f"Saved {out}.png / .pdf  (ymax {a.ymax})")


if __name__ == "__main__":
    main()
