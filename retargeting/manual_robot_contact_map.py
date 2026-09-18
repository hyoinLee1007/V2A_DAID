#!/usr/bin/env python3
"""로봇 손 표면에서 접촉시킬 부위를 직접 골라 contact map을 만든다.

``reconstruction/manual_contact_heatmap.py`` 가 물체 쪽에서 하는 일의 로봇 쪽
짝. 다만 이 머신에는 ``open3d`` 와 ``mujoco`` 를 함께 가진 환경이 없어서
(retargeting 에 mujoco, sam3d-objects 에 open3d) 세 단계로 나뉜다.

    # 1) retargeting 환경 — 손/물체 표면을 꺼낸다
    python manual_robot_contact_map.py export --run outputs/sharpa/left/cupmove_decay2/0

    # 2) sam3d-objects 환경 — 직접 고른다 (open3d 창)
    python manual_robot_contact_map.py pick

    # 3) retargeting 환경 — 고른 것을 리워드가 쓰는 형식으로 저장
    python manual_robot_contact_map.py build --out outputs/manual_robot_contact_map.npz

리워드가 필요한 건 정점 인덱스가 아니라 ``(body_id, 그 body 프레임의 offset)``
이다. 손이 움직여도 유효하고, 리워드는 매 스텝 ``xpos[b] + xmat[b] @ offset``
으로 현재 위치를 얻는다. build 단계가 그 변환을 한다.

로봇 손은 링크별로 분리된 메쉬라 표면을 가로지르는 geodesic 이 없다. 그래서
seed 주변 확산은 **같은 링크 안에서만** Euclidean 으로 한다 — 마디 경계를 넘지
않으므로 마디마다 따로 골라야 한다.

짝짓기는 ``--groups N`` 으로 정한다. 그룹마다 (손 부위, 물체 부위) 를 한 번씩
고르고, 그 그룹의 손 점들은 **그 그룹의 물체 지점만** 목표로 삼는다. 그룹이
하나면 손 점 전체가 물체 선택 전체를 공유하는 곱집합이 되는데, 그러면 각 점이
결국 가장 가까운 지점으로 가므로 두 손가락이 한 자리에 몰릴 수 있다 — 자동
파이프라인에서 짝을 복원해 고쳤던 바로 그 실패다. 검지와 중지에 서로 다른
지점을 지정하려면 그룹을 나눠야 한다.

조작:  Shift+왼쪽클릭 = seed 추가,  Shift+오른쪽클릭 = 취소,  Q = 완료
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_DIR = "outputs/manual_pick"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("export", "pick", "build"))
    p.add_argument("--dir", default=DEFAULT_DIR,
                   help="세 단계가 주고받는 중간 파일 디렉터리")
    p.add_argument("--run", default="outputs/sharpa/left/cupmove_decay2/0")
    p.add_argument("--scene", default=None)
    p.add_argument("--side", default="left", choices=("left", "right"))
    p.add_argument("--sigma", type=float, default=0.008,
                   help="손 seed 확산 거리(m), 같은 링크 안에서만. 기본 8 mm 는 "
                        "손가락 마디 폭 정도.")
    p.add_argument("--object-sigma", type=float, default=0.012)
    p.add_argument("--cutoff", type=float, default=2.5)
    p.add_argument("--max-points", type=int, default=0,
                   help="저장할 손 쪽 점 수 상한 (0 = 무제한). 리워드는 자체적으로 "
                        "robot_contact_max_points 로 다시 자른다.")
    p.add_argument("--no-object", action="store_true")
    p.add_argument("--groups", type=int, default=1,
                   help="(손 부위, 물체 부위) 짝의 개수. 그룹마다 창이 두 번 뜨고, "
                        "그 그룹의 손 점은 그 그룹의 물체 지점만 목표로 삼는다. "
                        "1이면 곱집합이 되어 손 점마다 가장 가까운 지점으로 간다.")
    p.add_argument("--point-size", type=float, default=6.0)
    p.add_argument("--out", default="outputs/manual_robot_contact_map.npz")
    return p.parse_args()


def do_export(a: argparse.Namespace) -> None:
    import mujoco
    from retargeting.utils.mano_robot_map import (
        _bid, _canonical, body_mesh_world, segments)
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts

    run = Path(a.run)
    scene = a.scene or str(run / "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene)
    data = _canonical(model, a.side)     # 열린 손: 링크가 겹치지 않아 고르기 쉽다

    seg_of_body = {}
    for seg in segments(a.side):
        for nm in list(seg["rob_surf"]) + [seg["rob_frame"]]:
            seg_of_body[_bid(model, nm)] = (seg["finger"], seg["name"])

    verts, owner = [], []
    for b in seg_of_body:
        v, _ = body_mesh_world(model, data, b)
        if v is not None and len(v):
            verts.append(v)
            owner.append(np.full(len(v), b, np.int64))
    if not verts:
        raise SystemExit(f"{a.side} 손 표면 메쉬를 찾지 못했습니다")
    P, B = np.concatenate(verts), np.concatenate(owner)

    ob = _bid(model, f"{a.side}_object")
    V = _visual_mesh_local_verts(model, a.side)
    Vw = V @ data.xmat[ob].reshape(3, 3).T + data.xpos[ob]

    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "surfaces.npz", hand=P, hand_body=B, obj=Vw,
             body_ids=np.array(sorted(seg_of_body)),
             body_finger=np.array([seg_of_body[b][0] for b in sorted(seg_of_body)]),
             body_segment=np.array([seg_of_body[b][1] for b in sorted(seg_of_body)]),
             side=a.side, scene=scene, run=str(run))
    print(f"손 표면 정점 {len(P)}개 ({len(seg_of_body)} 링크), 물체 정점 {len(Vw)}개")
    print(f"Saved {d / 'surfaces.npz'}")
    print(f"\n다음: sam3d-objects 환경에서")
    print(f"  python {Path(__file__).name} pick --dir {a.dir}")


def _pick(points: np.ndarray, title: str, lines: list[str], size: float) -> np.ndarray:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), 0.70))
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    for l in lines:
        print("  " + l)
    print("\n  Shift+왼쪽클릭 = seed 추가   Shift+오른쪽클릭 = 취소   Q = 완료")
    print("=" * 70)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=900)
    vis.add_geometry(pcd)
    ro = vis.get_render_option()
    if ro is not None:
        ro.point_size = float(size)
        ro.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.run()
    picked = np.unique(np.asarray(vis.get_picked_points(), dtype=np.int64))
    vis.destroy_window()
    if len(picked) == 0:
        raise SystemExit("선택된 seed가 없습니다. Shift 를 누른 채 클릭하세요.")
    print(f"[INFO] seed {len(picked)}개")
    return picked


def do_pick(a: argparse.Namespace) -> None:
    d = Path(a.dir)
    s = np.load(d / "surfaces.npz", allow_pickle=True)
    hand_groups, obj_groups = [], []
    for g in range(max(1, a.groups)):
        tag = f" [그룹 {g + 1}/{a.groups}]" if a.groups > 1 else ""
        hand_groups.append(_pick(
            np.asarray(s["hand"]), f"로봇 손: 접촉시킬 부위{tag}",
            ["원본 영상에서 사람이 물체에 닿았던 손의 부위를 고르세요.",
             f"seed 주변 {a.sigma*1000:.0f} mm 가 같은 링크 안에서 함께 잡힙니다.",
             "링크 경계는 넘지 않으니 마디마다 따로 고르세요."],
            a.point_size))
        if a.no_object:
            obj_groups.append(np.zeros(0, np.int64))
            continue
        obj_groups.append(_pick(
            np.asarray(s["obj"]), f"물체: 위 부위가 닿아야 할 지점{tag}",
            ["방금 고른 손 부위가 닿아야 할 물체 표면을 고르세요.",
             f"seed 주변 {a.object_sigma*1000:.0f} mm 가 함께 잡힙니다.",
             "이 그룹의 손 점은 여기서 고른 지점만 목표로 삼습니다."],
            a.point_size))
    # 길이가 제각각이라 flat + offset 으로 저장한다.
    def flat(gs):
        off = np.cumsum([0] + [len(x) for x in gs]).astype(np.int64)
        return (np.concatenate(gs) if gs else np.zeros(0, np.int64)), off
    hs, ho = flat(hand_groups)
    os_, oo = flat(obj_groups)
    np.savez(d / "picks.npz", hand_seed=hs, hand_group_offset=ho,
             object_seed=os_, object_group_offset=oo, groups=max(1, a.groups))
    print(f"\nSaved {d / 'picks.npz'}")
    print(f"\n다음: retargeting 환경에서")
    print(f"  python {Path(__file__).name} build --dir {a.dir} --out <경로>")


def do_build(a: argparse.Namespace) -> None:
    import mujoco
    d = Path(a.dir)
    s = np.load(d / "surfaces.npz", allow_pickle=True)
    pk = np.load(d / "picks.npz")
    P, B = np.asarray(s["hand"]), np.asarray(s["hand_body"])
    side, scene = str(s["side"]), str(s["scene"])
    bid = np.asarray(s["body_ids"])
    fin = {int(b): f for b, f in zip(bid, s["body_finger"].astype(str))}
    sg = {int(b): g for b, g in zip(bid, s["body_segment"].astype(str))}

    # 같은 링크 안에서만 확산. 분리된 메쉬라 geodesic 대신 Euclidean.
    # 그룹별로 따로 퍼뜨려서 어느 점이 어느 그룹 소속인지 유지한다.
    ho = np.asarray(pk["hand_group_offset"]) if "hand_group_offset" in pk.files \
        else np.array([0, len(pk["hand_seed"])])
    hs = np.asarray(pk["hand_seed"])
    w = np.zeros(len(P))
    grp = np.full(len(P), -1, np.int64)
    for g in range(len(ho) - 1):
        for i in hs[ho[g]:ho[g + 1]]:
            same = B == B[i]
            dist = np.linalg.norm(P[same] - P[i], axis=1)
            gg = np.exp(-(dist ** 2) / (2 * a.sigma ** 2))
            gg[dist > a.sigma * a.cutoff] = 0.0
            idx = np.flatnonzero(same)
            better = gg > w[idx]
            w[idx[better]] = gg[better]
            grp[idx[better]] = g
    keep = np.flatnonzero(w > 1e-3)
    if a.max_points > 0 and len(keep) > a.max_points:
        keep = keep[np.argsort(-w[keep])[:a.max_points]]
    kgrp = grp[keep]

    model = mujoco.MjModel.from_xml_path(a.scene or scene)
    from retargeting.utils.mano_robot_map import _bid, _canonical
    data = _canonical(model, side)
    body_id = B[keep]
    R = data.xmat[body_id].reshape(-1, 3, 3)
    local_offset = np.einsum("nji,nj->ni", R, P[keep] - data.xpos[body_id])
    finger = np.array([fin[int(b)] for b in body_id])
    segment = np.array([sg[int(b)] for b in body_id])
    body_name = np.array([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b))
                          for b in body_id])
    weight = w[keep]

    print(f"손 쪽 점 {len(keep)}개")
    print(f"\n{'finger':8s} {'pts':>5s} {'weight':>8s}   segments")
    for f in sorted(set(finger)):
        k = finger == f
        print(f"{f:8s} {int(k.sum()):5d} {weight[k].sum():8.2f}   "
              f"{' '.join(sorted(set(segment[k])))}")

    tgt_offset = np.zeros(len(keep) + 1, np.int64)
    tgt_vertex = np.zeros(0, np.int64)
    tgt_weight = np.zeros(0, np.float32)
    oseed = np.asarray(pk["object_seed"])
    oo = np.asarray(pk["object_group_offset"]) if "object_group_offset" in pk.files \
        else np.array([0, len(oseed)])
    if len(oseed):
        from retargeting.utils.contact_region_reward import _visual_mesh_local_verts
        V = _visual_mesh_local_verts(model, side)      # 물체 body 프레임
        # 그룹별 목표 집합. 손 점은 자기 그룹의 물체 지점만 목표로 삼는다.
        per_group = []
        for g in range(len(oo) - 1):
            ow = np.zeros(len(V))
            for i in oseed[oo[g]:oo[g + 1]]:
                dist = np.linalg.norm(V - V[i], axis=1)
                gg = np.exp(-(dist ** 2) / (2 * a.object_sigma ** 2))
                gg[dist > a.object_sigma * a.cutoff] = 0.0
                ow = np.maximum(ow, gg)
            ok = np.flatnonzero(ow > 1e-3)
            per_group.append((ok, ow[ok].astype(np.float32)))
            print(f"그룹 {g + 1} 목표 물체 정점 {len(ok)}개")
        pieces_v, pieces_w = [], []
        for n, k in enumerate(keep):
            g = int(kgrp[n]) if kgrp[n] >= 0 else 0
            g = min(g, len(per_group) - 1)
            pieces_v.append(per_group[g][0])
            pieces_w.append(per_group[g][1])
            tgt_offset[n + 1] = tgt_offset[n] + len(per_group[g][0])
        tgt_vertex = np.concatenate(pieces_v)
        tgt_weight = np.concatenate(pieces_w)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, body_id=body_id, local_offset=local_offset, weight=weight,
             finger=finger, segment=segment, body_name=body_name,
             tgt_offset=tgt_offset, tgt_vertex=tgt_vertex, tgt_weight=tgt_weight,
             hand_seed=pk["hand_seed"], object_seed=oseed,
             side=side, scene=scene, manual=True)
    print(f"\nSaved {out}")
    print("config 의 robot_contact_map_path 를 이 파일로 바꾸면 리워드가 씁니다.")


def main() -> None:
    a = parse_args()
    {"export": do_export, "pick": do_pick, "build": do_build}[a.stage](a)


if __name__ == "__main__":
    main()
