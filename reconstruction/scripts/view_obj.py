#!/usr/bin/env python3
"""OBJ/PLY 메쉬를 검사하고 GLB로 내보낸다.

재구성이 만든 물체 메쉬가 이상해 보일 때, 눈으로 보기 전에 숫자가 먼저 말해
주는 것들이 있다 — 연결 요소가 여러 개면 조각난 것이고, 면 수 대비 정점 수가
이상하면 중복 정점이며, bbox 한 축이 유독 얇으면 납작하게 재구성된 것이다.

    python view_obj.py <mesh> [<mesh> ...]
    python view_obj.py dataset/sliceKnife/video_segmentation/masks/frame_000004_masks/knife/knife.obj

여러 개를 주면 나란히 배치해 한 GLB로 내보내므로 크기를 서로 비교할 수 있다.
정점 순서를 보존해야 하므로 mesh loader 의 병합/재정렬을 끄고 읽는다
(``process=False``) — 히트맵이 정점 인덱스로 색인되기 때문이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def load_obj_text(path: str):
    """정점 순서를 파일 그대로 유지하며 읽는다."""
    v, f = [], []
    for line in open(path, errors="ignore"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [t.split("/")[0] for t in line.split()[1:]]
            idx = [int(i) - 1 if int(i) > 0 else len(v) + int(i) for i in idx]
            for k in range(1, len(idx) - 1):          # 다각형은 삼각형으로
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(v, float), np.asarray(f, np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", nargs="+")
    p.add_argument("--out", default="outputs/view_obj.glb")
    p.add_argument("--scale", type=float, default=1.0,
                   help="내보내기 전에 곱할 배율 (정규화 좌표를 미터로 볼 때).")
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    scene = trimesh.Scene()
    x = 0.0
    for path in args.mesh:
        if path.lower().endswith(".obj"):
            V, F = load_obj_text(path)
        else:
            m = trimesh.load(path, process=False, force="mesh")
            V, F = np.asarray(m.vertices), np.asarray(m.faces)
        V = V * args.scale
        m = trimesh.Trimesh(V, F, process=False)
        comps = m.split(only_watertight=False)
        dup = len(V) - len(np.unique(np.round(V, 9), axis=0))
        print(f"\n{path}")
        print(f"  정점 {len(V)}  면 {len(F)}   중복 정점 {dup}")
        print(f"  bbox (cm) {np.round(np.ptp(V, 0) * 100, 2)}   "
              f"대각선 {np.linalg.norm(np.ptp(V, 0)) * 100:.2f} cm")
        print(f"  watertight {m.is_watertight}   winding 일관 {m.is_winding_consistent}   "
              f"연결 요소 {len(comps)}   euler {m.euler_number}")
        if len(comps) > 1:
            sz = sorted((len(c.vertices) for c in comps), reverse=True)
            print(f"    요소 크기 상위: {sz[:6]}"
                  + ("  ← 조각남" if sz[1] > len(V) * 0.02 else "  (나머지는 부스러기)"))
        deg = m.area_faces
        if len(deg):
            tiny = int((deg < deg.mean() * 1e-4).sum())
            print(f"  면적 0에 가까운 면 {tiny}개")
        shift = np.array([x - V[:, 0].min(), 0.0, 0.0]) - np.array(
            [0.0, V[:, 1].mean(), V[:, 2].mean()])
        mm = trimesh.Trimesh(V + shift, F, process=False)
        mm.visual = trimesh.visual.ColorVisuals(
            mesh=mm, vertex_colors=np.tile([200, 195, 185, 255], (len(V), 1)))
        scene.add_geometry(mm)
        x += np.ptp(V, 0)[0] * 1.3

    if not args.no_export:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        scene.export(out)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
