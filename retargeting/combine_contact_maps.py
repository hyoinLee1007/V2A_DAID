#!/usr/bin/env python3
"""자동 추출한 손 맵에, 사람이 지정한 물체 맵을 목표로 붙인다.

짝 기반 인력항(robot_contact_paired)은 물체 히트맵을 직접 읽지 않는다. 로봇 맵
안의 점별 목표 리스트(tgt_vertex)를 읽고, 그 리스트는 extract_contact_map.py 가
자동 추출한 (손 정점, 물체 정점) 짝에서 온다. 그래서 config 의
contact_heatmap_path 만 수동 맵으로 바꾸면 수동 맵은 **무시되고** 자동 짝이 그대로
쓰인다. 비-paired 경로는 물체 히트맵을 쓰지만 점별 비용을 합산하므로 lambda_c 의
정규화가 깨진다.

이 스크립트는 손 쪽(어느 스킨점이, 어떤 가중치로)은 자동 맵에서 그대로 가져오고,
목표만 수동 물체 맵의 고신뢰 영역으로 바꾼다. 가중평균 경로를 유지하므로
lambda_c 해석이 그대로 성립한다.

두 방식:
  --mode all        모든 스킨점이 수동 영역 전체를 목표로 삼는다 (기본).
                    자동 짝이 물체의 엉뚱한 곳을 가리켰더라도 영향받지 않는다.
  --mode intersect  각 스킨점의 자동 목표 중 수동 영역에 든 것만 남긴다.
                    시연의 손가락별 대응을 보존하지만, 자동 짝이 잘못된 곳을
                    가리켰다면 목표가 비어 그 점은 빠진다.

    python combine_contact_maps.py \
        --hand-map outputs/robot_contact_map_lotion_index.npz \
        --object-heatmap ../reconstruction/heatmap_out/contact_heatmap_lotion_manual_objorder.npz \
        --scene outputs/sharpa/right/lotion_index/0/scene.xml --side right \
        --out outputs/robot_contact_map_lotion_index_manual.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import mujoco
    from retargeting.utils.contact_region_reward import _visual_mesh_local_verts

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand-map", required=True, help="build_robot_contact_map.py 출력")
    p.add_argument("--object-heatmap", required=True,
                   help="수동 물체 맵. fix_heatmap_vertex_order.py 를 거쳐 씬 메쉬와 "
                        "정점 순서가 같아야 한다.")
    p.add_argument("--scene", required=True)
    p.add_argument("--side", required=True, choices=("left", "right"))
    p.add_argument("--thresh", type=float, default=0.5, help="목표로 쓸 신뢰도 기준")
    p.add_argument("--mode", default="all", choices=("all", "intersect"))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    hm = np.load(a.hand_map, allow_pickle=True)
    H = np.asarray(np.load(a.object_heatmap)["contact_confidence"], float)
    m = mujoco.MjModel.from_xml_path(a.scene)
    V = _visual_mesh_local_verts(m, a.side)
    if len(H) != len(V):
        raise SystemExit(
            f"수동 맵 {len(H)} 정점 vs 씬 {a.side}_visual {len(V)} 정점 — 다른 메쉬이거나 "
            "fix_heatmap_vertex_order.py 를 안 거쳤습니다")

    hot = np.flatnonzero(H > a.thresh)
    if len(hot) == 0:
        raise SystemExit(f"H > {a.thresh} 인 정점이 없습니다")
    n = len(hm["body_id"])
    print(f"손 맵: 스킨점 {n}개  (자동 추출)")
    print(f"물체 맵: H>{a.thresh} 정점 {len(hot)} / {len(V)}  (수동)")

    old_off = np.asarray(hm["tgt_offset"], np.int64)
    old_v = np.asarray(hm["tgt_vertex"], np.int64)
    hot_set = set(hot.tolist())
    pieces_v, pieces_w = [], []
    offset = np.zeros(n + 1, np.int64)
    empty = 0
    for k in range(n):
        if a.mode == "all":
            v = hot
        else:
            v = np.array([x for x in old_v[old_off[k]:old_off[k + 1]] if x in hot_set],
                         np.int64)
            if len(v) == 0:
                empty += 1
        pieces_v.append(v)
        pieces_w.append(H[v].astype(np.float32))
        offset[k + 1] = offset[k] + len(v)

    out = {k: hm[k] for k in hm.files if not k.startswith("tgt_")}
    out.update(tgt_offset=offset,
               tgt_vertex=np.concatenate(pieces_v) if pieces_v else np.zeros(0, np.int64),
               tgt_weight=np.concatenate(pieces_w) if pieces_w else np.zeros(0, np.float32),
               object_source=str(a.object_heatmap), target_mode=a.mode,
               target_thresh=a.thresh)
    np.savez(a.out, **out)
    counts = np.diff(offset)
    print(f"\n목표 리스트 ({a.mode}): 점당 중앙값 {int(np.median(counts))}개")
    if empty:
        print(f"  [!] 자동 목표가 수동 영역과 겹치지 않아 빈 점 {empty}/{n}개 — "
              "자동 짝이 물체의 다른 곳을 가리켰다는 뜻입니다. --mode all 을 고려하세요.")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
