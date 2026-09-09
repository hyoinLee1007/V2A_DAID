#!/usr/bin/env python3
'''python visualize_contact_heatmap.py \
    --heatmap "/절대경로/heatmap_out/contact_heatmap.npz" \
    --show-seeds 
    '''
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contact heatmap을 3D mesh 위에 시각화"
    )

    parser.add_argument(
        "--heatmap",
        required=True,
        help="contact_heatmap.npz 절대/상대 경로"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="시각화용 최소 contact confidence (기본값: 0.0)"
    )

    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="수동 annotation seed point 표시"
    )

    return parser.parse_args()


def load_heatmap(path):
    data = np.load(path)

    required = [
        "vertices",
        "faces",
        "contact_confidence",
    ]

    for key in required:
        if key not in data:
            raise KeyError(
                f"{path}에 '{key}' 데이터가 없습니다."
            )

    vertices = data["vertices"]
    faces = data["faces"]
    heatmap = data["contact_confidence"]

    seeds = None

    if "seed_vertex_indices" in data:
        seeds = data["seed_vertex_indices"]

    if len(vertices) != len(heatmap):
        raise ValueError(
            f"vertex 수({len(vertices)})와 "
            f"heatmap 수({len(heatmap)})가 다릅니다."
        )

    return vertices, faces, heatmap, seeds


def create_colored_mesh(
    vertices,
    faces,
    heatmap,
    threshold=0.0
):
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(
        vertices.astype(np.float64)
    )

    mesh.triangles = o3d.utility.Vector3iVector(
        faces.astype(np.int32)
    )

    # 0~1 범위 제한
    H = np.clip(heatmap, 0.0, 1.0)

    # threshold 이하를 0으로 표시
    if threshold > 0:
        H = H.copy()
        H[H < threshold] = 0.0

    # turbo:
    # 0 근처 -> 차가운 색
    # 1 근처 -> 뜨거운 색
    cmap = plt.colormaps["turbo"]

    colors = cmap(H)[:, :3]

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64)
    )

    mesh.compute_vertex_normals()

    return mesh


def create_seed_points(vertices, seeds):
    if seeds is None or len(seeds) == 0:
        return None

    valid_seeds = seeds[
        (seeds >= 0) &
        (seeds < len(vertices))
    ]

    if len(valid_seeds) == 0:
        return None

    points = vertices[valid_seeds]

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )

    # annotation seed는 검은색으로 표시
    colors = np.zeros(
        (len(points), 3),
        dtype=np.float64
    )

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def print_statistics(heatmap):
    print("\n===== Heatmap 통계 =====")
    print(f"vertex 개수 : {len(heatmap)}")
    print(f"최솟값      : {heatmap.min():.4f}")
    print(f"최댓값      : {heatmap.max():.4f}")
    print(f"평균        : {heatmap.mean():.4f}")

    for threshold in [0.25, 0.5, 0.75, 0.9]:
        count = np.sum(heatmap >= threshold)

        print(
            f"H >= {threshold:.2f} : "
            f"{count:6d} vertices "
            f"({count / len(heatmap) * 100:.2f}%)"
        )


def main():
    args = parse_args()

    heatmap_path = Path(
        args.heatmap
    ).expanduser().resolve()

    if not heatmap_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {heatmap_path}"
        )

    vertices, faces, heatmap, seeds = load_heatmap(
        heatmap_path
    )

    print_statistics(heatmap)

    mesh = create_colored_mesh(
        vertices,
        faces,
        heatmap,
        threshold=args.threshold
    )

    geometries = [mesh]

    if args.show_seeds:
        seed_points = create_seed_points(
            vertices,
            seeds
        )

        if seed_points is not None:
            geometries.append(seed_points)

            print(
                f"\n수동 annotation seed: "
                f"{len(seeds)}개"
            )

    print("""
===== 색상 의미 =====

파랑 계열 : contact confidence 낮음
초록/노랑 : 중간
빨강 계열 : contact confidence 높음

마우스 왼쪽 드래그 : 회전
휠                  : 확대/축소
Shift + 드래그      : 이동
Q                   : 종료
=====================
""")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Contact Heatmap",
        width=1280,
        height=900,
        mesh_show_back_face=True
    )


if __name__ == "__main__":
    main()
