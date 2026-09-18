#!/usr/bin/env python3
"""접촉 맵이 의도한 접촉을 실제로 만들었는가 — 논문 표용 지표.

두 축을 따로 잰다. 물체 쪽 지표는 히트맵만 있으면 되므로 베이스라인에도
적용되고, 손 쪽 지표는 로봇 접촉 맵이 지정한 살갗 패치가 자기 목표에
닿았는지를 본다.

  desired contact error   접촉이 의도한 곳에서 얼마나 벗어났는가 (낮을수록 좋음)
  contact coverage        의도한 접촉 중 실제로 이뤄진 비율 (높을수록 좋음)

둘을 함께 봐야 하는 이유: 오차만 보면 "한 점만 정확히 닿고 나머지는 안 닿음"
이 만점을 받고, coverage 만 보면 "넓게 대충 닿음" 이 만점을 받는다.

    python contact_metrics.py --run outputs/sharpa/left/cupmove/0 \
        --heatmap ../reconstruction/heatmap_out/contact_heatmap_palmar.npz \
        --map outputs/robot_contact_map_left.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import mujoco, torch
    from scipy.spatial import cKDTree
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    from retargeting.utils.mano_robot_map import _bid

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--scene", default=None)
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--heatmap", required=True)
    p.add_argument("--map", default=None,
                   help="robot_contact_map npz. 없으면 물체 쪽 지표만 낸다.")
    p.add_argument("--thresh", type=float, default=0.5,
                   help="'의도한 접촉 영역' 으로 볼 히트맵 신뢰도 기준.")
    p.add_argument("--radius", type=float, default=0.01,
                   help="coverage 를 셀 때 '닿았다' 로 보는 반경 (m).")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--label", default=None)
    a = p.parse_args()

    run = Path(a.run)
    m = mujoco.MjModel.from_xml_path(a.scene or str(run / "scene.xml"))
    d = mujoco.MjData(m)
    ob = _bid(m, f"{a.side}_object")
    V = _visual_mesh_local_verts(m, a.side)
    H = np.asarray(np.load(a.heatmap)["contact_confidence"], float)
    if len(H) != len(V):
        raise SystemExit(f"히트맵 {len(H)} vs {a.side}_visual {len(V)} — 다른 물체")
    hot = np.flatnonzero(H > a.thresh)
    tree_all, tree_hot = cKDTree(V), cKDTree(V[hot])

    t = np.load(run / "trajectory_mjwp.npz", allow_pickle=True)
    q = np.asarray(t["qpos"], float).reshape(-1, m.nq)
    w = np.asarray(t["warmup_progress"], float).reshape(-1)
    we = int(np.argmax(w >= 1.0)) * (len(q) // len(w)) if (w >= 1.0).any() else 0

    is_hand = np.array([b != ob and b != 0 for b in m.geom_bodyid])
    is_obj = m.geom_bodyid == ob

    # 손 쪽 맵 (선택)
    body = off = wt = None
    tgt_xyz = tgt_off = tgt_empty = None
    if a.map:
        from retargeting.utils.paired_contact_cost import build_targets
        cm = np.load(a.map, allow_pickle=True)
        wa = np.asarray(cm["weight"], float)
        cap = 64
        keep = np.argpartition(-wa, cap)[:cap] if len(wa) > cap else np.arange(len(wa))
        body = cm["body_id"].astype(np.int64)[keep]
        off = np.asarray(cm["local_offset"], float)[keep]
        wt = torch.tensor(wa[keep])
        built = build_targets(cm["tgt_offset"], cm["tgt_vertex"], cm["tgt_weight"],
                              keep, V, 0.0, "cpu", torch.float64)
        if built is not None:
            tgt_xyz, tgt_off, tgt_empty = built

    err_obj, hit_idx, per_pt_min = [], set(), None
    for s in range(we, len(q), a.stride):
        d.qpos[:] = q[s]
        mujoco.mj_forward(m, d)
        R = d.xmat[ob].reshape(3, 3)
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = con.geom1, con.geom2
            if not ((is_hand[g1] and is_obj[g2]) or (is_obj[g1] and is_hand[g2])):
                continue
            loc = (con.pos - d.xpos[ob]) @ R          # 물체 로컬로
            dist, j = tree_hot.query(loc)
            err_obj.append(dist)
            if dist < a.radius:
                hit_idx.add(int(hot[j]))
        if tgt_xyz is not None:
            P = d.xpos[body] + np.einsum("nij,nj->ni",
                                         d.xmat[body].reshape(-1, 3, 3), off)
            loc = torch.tensor((P - d.xpos[ob]) @ R)[None]
            pp = (loc * loc).sum(-1, keepdim=True)
            xx = (tgt_xyz * tgt_xyz).sum(-1)[None]
            px = torch.einsum("wkd,ktd->wkt", loc, tgt_xyz)
            c_k = ((pp + xx - 2 * px).clamp_min(0).sqrt() + tgt_off[None]).amin(-1)[0].numpy()
            per_pt_min = c_k if per_pt_min is None else np.minimum(per_pt_min, c_k)

    err_obj = np.asarray(err_obj)
    name = a.label or run.parent.name
    print(f"\n=== {name}")
    print(f"의도한 접촉 영역: H>{a.thresh} 인 물체 정점 {len(hot)} / {len(V)}")
    if len(err_obj) == 0:
        print("  손-물체 접촉 없음"); return
    print(f"\n[물체 쪽]  손-물체 접촉 {len(err_obj)}개")
    print(f"  desired contact error   평균 {err_obj.mean()*100:6.2f} cm   "
          f"중앙 {np.median(err_obj)*100:6.2f} cm   p90 {np.percentile(err_obj,90)*100:6.2f} cm")
    print(f"  contact precision       {100*(err_obj < a.radius).mean():5.1f}% "
          f"(접촉이 의도 영역 {a.radius*100:.0f} cm 이내)")
    print(f"  contact coverage        {100*len(hit_idx)/max(len(hot),1):5.1f}% "
          f"({len(hit_idx)} / {len(hot)} 의도 정점이 닿음)")
    if per_pt_min is not None:
        live = ~tgt_empty.numpy()
        pm = per_pt_min[live]
        print(f"\n[손 쪽]  매핑 스킨점 {int(live.sum())}개")
        print(f"  desired contact error   평균 {pm.mean()*100:6.2f} cm   "
              f"중앙 {np.median(pm)*100:6.2f} cm")
        print(f"  contact coverage        {100*(pm < a.radius).mean():5.1f}% "
              f"(자기 목표 {a.radius*100:.0f} cm 이내에 도달한 점)")


if __name__ == "__main__":
    main()
