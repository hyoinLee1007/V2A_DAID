#!/usr/bin/env python3
"""접촉이 히트맵이 가리킨 영역에 실제로 떨어졌는가.

``analyze_grasp_contacts.py`` 는 로봇 접촉 맵(손 쪽 대응)을 기준으로 재므로,
물체 히트맵만 준 실험에는 쓸 수 없다. 이것은 물체 쪽만 본다: 손-물체 접촉점
각각에 대해 그 지점의 히트맵 신뢰도와, 고신뢰 영역까지의 거리를 잰다.

    python analyze_heatmap_hit.py --run outputs/sharpa/right/metalcupmove_heat300/0 \
        --side right --heatmap ../reconstruction/heatmap_out/contact_heatmap_metalcup_objorder.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import mujoco
    from scipy.spatial import cKDTree
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
    from retargeting.utils.mano_robot_map import _bid

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--scene", default=None)
    p.add_argument("--side", default="right", choices=("left", "right"))
    p.add_argument("--heatmap", required=True)
    p.add_argument("--thresh", type=float, default=0.5,
                   help="'지정된 영역' 으로 볼 신뢰도 기준.")
    p.add_argument("--stride", type=int, default=5)
    a = p.parse_args()

    run = Path(a.run)
    m = mujoco.MjModel.from_xml_path(a.scene or str(run / "scene.xml"))
    d = mujoco.MjData(m)
    ob = _bid(m, f"{a.side}_object")
    V = _visual_mesh_local_verts(m, a.side)
    H = np.asarray(np.load(a.heatmap)["contact_confidence"], float)
    if len(H) != len(V):
        raise SystemExit(f"히트맵 {len(H)} 정점 vs {a.side}_visual {len(V)} — 다른 물체")
    hot = np.flatnonzero(H > a.thresh)
    tree_all, tree_hot = cKDTree(V), cKDTree(V[hot])
    print(f"히트맵: H>{a.thresh} 인 정점 {len(hot)} / {len(V)}")

    is_hand = np.array([b != ob and b != 0 for b in m.geom_bodyid])
    is_obj = m.geom_bodyid == ob
    t = np.load(run / "trajectory_mjwp.npz", allow_pickle=True)
    q = np.asarray(t["qpos"], float).reshape(-1, m.nq)
    w = np.asarray(t["warmup_progress"], float).reshape(-1)
    we = int(np.argmax(w >= 1.0)) * (len(q) // len(w)) if (w >= 1.0).any() else 0

    conf, dist = [], []
    for s in range(we, len(q), a.stride):
        d.qpos[:] = q[s]
        mujoco.mj_forward(m, d)
        R = d.xmat[ob].reshape(3, 3)
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = con.geom1, con.geom2
            if not ((is_hand[g1] and is_obj[g2]) or (is_obj[g1] and is_hand[g2])):
                continue
            # 접촉점을 물체 로컬 프레임으로 옮겨 히트맵과 같은 좌표에서 본다
            loc = (con.pos - d.xpos[ob]) @ R
            _, j = tree_all.query(loc)
            conf.append(H[j])
            dist.append(tree_hot.query(loc)[0])
    if not conf:
        raise SystemExit("손-물체 접촉이 없습니다")
    conf, dist = np.asarray(conf), np.asarray(dist)
    print(f"\n손-물체 접촉 {len(conf)}개")
    print(f"  접촉 지점의 히트맵 신뢰도: 평균 {conf.mean():.3f}  중앙 {np.median(conf):.3f}")
    print(f"    H>{a.thresh} 인 지점에 떨어진 접촉: {100*(conf>a.thresh).mean():5.1f}%")
    print(f"    H>0    인 지점에 떨어진 접촉: {100*(conf>0).mean():5.1f}%")
    print(f"  지정 영역까지 거리: 평균 {dist.mean()*100:5.2f} cm  "
          f"중앙 {np.median(dist)*100:5.2f} cm  p90 {np.percentile(dist,90)*100:5.2f} cm")
    for r in (0.005, 0.01, 0.02):
        print(f"    {r*100:.0f} cm 이내: {100*(dist<r).mean():5.1f}%")


if __name__ == "__main__":
    main()
