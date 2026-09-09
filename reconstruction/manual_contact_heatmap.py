#!/usr/bin/env python3
"""
컵 mesh에서 사람이 접촉했다고 판단한 영역을 수동으로 선택하고,
mesh surface geodesic distance 기반 continuous contact heatmap H(v) ∈ [0,1]을 생성한다.

실행 예:
    python manual_contact_heatmap.py \
        --mesh "/절대경로/cup.obj" \
        --out-dir "./heatmap_out"

선택 방법:
    Shift + 왼쪽 클릭   : contact seed 추가
    Shift + 오른쪽 클릭 : 마지막 선택 취소
    Q                   : 선택 완료

출력:
    contact_heatmap.npz
    contact_heatmap.json
    contact_heatmap_colored.ply
    contact_seed_vertices.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

try:
    from matplotlib import colormaps
except ImportError:
    colormaps = None


# ============================================================
# Argument
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Object Contact Heatmap Manual Annotation"
    )

    parser.add_argument(
        "--mesh",
        required=True,
        help="입력 OBJ/PLY mesh 경로"
    )

    parser.add_argument(
        "--out-dir",
        default="heatmap_out",
        help="결과 저장 디렉터리"
    )

    parser.add_argument(
        "--sigma-ratio",
        type=float,
        default=0.04,
        help=(
            "Gaussian sigma를 mesh bounding-box diagonal에 대한 비율로 지정. "
            "기본값: 0.04"
        )
    )

    parser.add_argument(
        "--cutoff-sigma",
        type=float,
        default=3.0,
        help=(
            "seed로부터 sigma * cutoff 이상 떨어진 영역의 heat를 0으로 설정. "
            "기본값: 3.0"
        )
    )

    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="참고용 binary contact threshold. 기본값: 0.5"
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="annotation 화면에서 vertex 크기. 기본값: 5"
    )

    return parser.parse_args()


# ============================================================
# Mesh loading
# ============================================================

def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:

    print(f"[INFO] Mesh loading: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(
        str(mesh_path),
        enable_post_processing=False
    )

    if mesh.is_empty():
        raise RuntimeError(
            f"Mesh를 읽을 수 없습니다: {mesh_path}"
        )

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0:
        raise RuntimeError("Mesh vertex가 없습니다.")

    if len(triangles) == 0:
        raise RuntimeError(
            "Triangle face가 없습니다. "
            "OBJ가 triangle mesh인지 확인하세요."
        )

    mesh.compute_vertex_normals()

    print(f"[INFO] Vertices : {len(vertices)}")
    print(f"[INFO] Faces    : {len(triangles)}")

    return mesh


# ============================================================
# Manual annotation
# ============================================================

def pick_contact_vertices(
    mesh: o3d.geometry.TriangleMesh,
    point_size: float
) -> np.ndarray:

    vertices = np.asarray(mesh.vertices)

    # --------------------------------------------------------
    # 중요:
    # TriangleMesh 자체를 picking하지 않고
    # 동일한 vertex 좌표를 가진 PointCloud를 생성
    # --------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        vertices.astype(np.float64)
    )

    # 밝은 회색
    colors = np.full(
        (len(vertices), 3),
        0.70,
        dtype=np.float64
    )

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print()
    print("=" * 70)
    print("Contact annotation")
    print("=" * 70)
    print()
    print("원본 RGB 영상을 보면서 사람이 실제로 잡았던")
    print("컵 표면의 위치를 선택하세요.")
    print()
    print("조작 방법")
    print()
    print("  Shift + 왼쪽 클릭   : Contact seed 추가")
    print("  Shift + 오른쪽 클릭 : 마지막 seed 삭제")
    print("  마우스 드래그        : 시점 회전")
    print("  휠                   : 확대 / 축소")
    print("  Q                    : annotation 종료")
    print()
    print("권장:")
    print("  실제 접촉 영역에 5~20개 정도 선택")
    print()
    print("=" * 70)

    vis = o3d.visualization.VisualizerWithEditing()

    vis.create_window(
        window_name="Manual Contact Annotation",
        width=1280,
        height=900
    )

    vis.add_geometry(pcd)

    # point를 좀 더 크게 표시
    render_option = vis.get_render_option()

    if render_option is not None:
        render_option.point_size = float(point_size)

        # 흰 배경
        render_option.background_color = np.asarray(
            [1.0, 1.0, 1.0]
        )

    vis.run()

    picked = vis.get_picked_points()

    vis.destroy_window()

    picked = np.asarray(
        picked,
        dtype=np.int64
    )

    picked = np.unique(picked)

    if len(picked) == 0:
        raise RuntimeError(
            "\n선택된 contact seed가 없습니다.\n"
            "반드시 Shift 키를 누른 상태에서 "
            "왼쪽 클릭하세요."
        )

    print()
    print("[INFO] 선택 완료")
    print(f"[INFO] Seed 개수: {len(picked)}")
    print(f"[INFO] Vertex indices:")
    print(picked.tolist())

    return picked


# ============================================================
# Mesh graph
# ============================================================

def build_surface_graph(
    vertices: np.ndarray,
    triangles: np.ndarray
):

    """
    Triangle mesh의 edge를 graph로 변환.

    각 edge weight:
        두 vertex 사이의 Euclidean distance

    이후 shortest path를 구하면
    mesh 표면을 따라가는 근사 geodesic distance가 된다.
    """

    triangles = triangles.astype(np.int64)

    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )

    # undirected graph
    edges_reverse = edges[:, ::-1]

    edges = np.vstack(
        [
            edges,
            edges_reverse
        ]
    )

    # 중복 제거
    edges = np.unique(
        edges,
        axis=0
    )

    src = edges[:, 0]
    dst = edges[:, 1]

    weights = np.linalg.norm(
        vertices[src] - vertices[dst],
        axis=1
    )

    graph = coo_matrix(
        (
            weights,
            (
                src,
                dst
            )
        ),
        shape=(
            len(vertices),
            len(vertices)
        )
    )

    return graph.tocsr()


# ============================================================
# Heatmap
# ============================================================

def create_contact_heatmap(
    vertices: np.ndarray,
    triangles: np.ndarray,
    seed_indices: np.ndarray,
    sigma_ratio: float,
    cutoff_sigma: float
):

    # mesh 전체 크기
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    bbox_diagonal = np.linalg.norm(
        bbox_max - bbox_min
    )

    sigma = bbox_diagonal * sigma_ratio

    if sigma <= 0:
        raise RuntimeError(
            "sigma 값이 0 이하입니다."
        )

    print()
    print("[INFO] Surface graph 생성 중...")

    graph = build_surface_graph(
        vertices,
        triangles
    )

    print("[INFO] Geodesic distance 계산 중...")

    # 각 seed에서 모든 vertex까지의 shortest path
    distance_matrix = dijkstra(
        graph,
        directed=False,
        indices=seed_indices,
        return_predecessors=False
    )

    if distance_matrix.ndim == 1:
        distance_matrix = distance_matrix[None, :]

    # 여러 seed 중 가장 가까운 seed
    min_distance = np.min(
        distance_matrix,
        axis=0
    )

    # --------------------------------------------------------
    # Gaussian contact heatmap
    #
    # H(v) = exp(-d(v,S)^2 / (2 sigma^2))
    # --------------------------------------------------------

    H = np.exp(
        -0.5 *
        (
            min_distance / sigma
        ) ** 2
    )

    # graph 연결이 안 된 vertex
    H[
        ~np.isfinite(min_distance)
    ] = 0.0

    # 너무 멀리 퍼지는 것 방지
    if cutoff_sigma > 0:

        cutoff_distance = (
            cutoff_sigma * sigma
        )

        H[
            min_distance > cutoff_distance
        ] = 0.0

    # 직접 선택한 seed는 반드시 1
    H[seed_indices] = 1.0

    return (
        H.astype(np.float32),
        min_distance.astype(np.float32),
        float(sigma),
        float(bbox_diagonal)
    )


# ============================================================
# Heatmap color
# ============================================================

def heatmap_to_color(
    heatmap: np.ndarray
):

    H = np.clip(
        heatmap,
        0.0,
        1.0
    )

    if colormaps is not None:

        # turbo:
        # 낮음 -> 파랑
        # 중간 -> 초록/노랑
        # 높음 -> 빨강
        colors = colormaps["turbo"](H)[:, :3]

    else:

        # matplotlib이 없을 경우 간단한 fallback
        colors = np.zeros(
            (len(H), 3),
            dtype=np.float64
        )

        colors[:, 0] = H
        colors[:, 2] = 1.0 - H

    return colors.astype(np.float64)


# ============================================================
# Save
# ============================================================

def save_results(
    mesh: o3d.geometry.TriangleMesh,
    mesh_path: Path,
    output_dir: Path,
    seed_indices: np.ndarray,
    H: np.ndarray,
    distance: np.ndarray,
    sigma: float,
    bbox_diagonal: float,
    binary_threshold: float
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    vertices = np.asarray(
        mesh.vertices
    ).astype(np.float32)

    faces = np.asarray(
        mesh.triangles
    ).astype(np.int32)

    binary_contact = (
        H >= binary_threshold
    ).astype(np.uint8)

    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    npz_path = (
        output_dir /
        "contact_heatmap.npz"
    )

    np.savez_compressed(
        npz_path,

        vertices=vertices,
        faces=faces,

        contact_confidence=H,

        geodesic_distance=distance,

        seed_vertex_indices=(
            seed_indices.astype(np.int64)
        ),

        binary_contact=binary_contact,

        sigma=np.float32(sigma),

        bbox_diagonal=np.float32(
            bbox_diagonal
        )
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        output_dir /
        "contact_heatmap.json"
    )

    json_data = {

        "source_mesh":
            str(mesh_path),

        "vertex_count":
            int(len(vertices)),

        "seed_vertex_indices":
            seed_indices.tolist(),

        "sigma":
            sigma,

        "bbox_diagonal":
            bbox_diagonal,

        "binary_threshold":
            binary_threshold,

        "contact_confidence":
            H.tolist()
    }

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as fp:

        json.dump(
            json_data,
            fp,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # seed indices
    # --------------------------------------------------------

    seed_path = (
        output_dir /
        "contact_seed_vertices.txt"
    )

    np.savetxt(
        seed_path,
        seed_indices,
        fmt="%d"
    )

    # --------------------------------------------------------
    # Colored PLY
    # --------------------------------------------------------

    colored_mesh = o3d.geometry.TriangleMesh(
        mesh
    )

    colors = heatmap_to_color(H)

    colored_mesh.vertex_colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    colored_mesh.compute_vertex_normals()

    colored_path = (
        output_dir /
        "contact_heatmap_colored.ply"
    )

    result = (
        o3d.io.write_triangle_mesh(
            str(colored_path),
            colored_mesh,
            write_vertex_colors=True,
            write_vertex_normals=True
        )
    )

    if not result:
        raise RuntimeError(
            "Colored mesh 저장에 실패했습니다."
        )

    return (
        colored_mesh,
        npz_path,
        json_path,
        colored_path,
        seed_path
    )


# ============================================================
# Seed visualization
# ============================================================

def create_seed_visualization(
    vertices,
    seed_indices,
    bbox_diagonal
):

    """
    seed point를 작은 구로 시각화한다.
    """

    geometries = []

    radius = bbox_diagonal * 0.008

    for idx in seed_indices:

        sphere = (
            o3d.geometry.TriangleMesh
            .create_sphere(
                radius=radius,
                resolution=10
            )
        )

        sphere.translate(
            vertices[idx]
        )

        # 검은색 seed marker
        sphere.paint_uniform_color(
            [0.0, 0.0, 0.0]
        )

        sphere.compute_vertex_normals()

        geometries.append(
            sphere
        )

    return geometries


# ============================================================
# Statistics
# ============================================================

def print_statistics(
    H,
    seed_indices,
    sigma,
    bbox_diagonal
):

    print()
    print("=" * 70)
    print("Heatmap statistics")
    print("=" * 70)

    print(
        f"Vertex count      : {len(H)}"
    )

    print(
        f"Seed count        : {len(seed_indices)}"
    )

    print(
        f"Bounding box diag : {bbox_diagonal:.6f}"
    )

    print(
        f"Sigma             : {sigma:.6f}"
    )

    print(
        f"Heatmap min       : {H.min():.6f}"
    )

    print(
        f"Heatmap max       : {H.max():.6f}"
    )

    print(
        f"Heatmap mean      : {H.mean():.6f}"
    )

    print()

    for threshold in [
        0.25,
        0.50,
        0.75,
        0.90
    ]:

        count = int(
            np.sum(
                H >= threshold
            )
        )

        ratio = (
            count /
            len(H) *
            100.0
        )

        print(
            f"H >= {threshold:.2f} : "
            f"{count:6d} vertices "
            f"({ratio:.2f}%)"
        )

    print("=" * 70)


# ============================================================
# Result visualization
# ============================================================

def visualize_result(
    colored_mesh,
    vertices,
    seed_indices,
    bbox_diagonal
):

    seed_geometries = (
        create_seed_visualization(
            vertices,
            seed_indices,
            bbox_diagonal
        )
    )

    geometries = [
        colored_mesh,
        *seed_geometries
    ]

    print()
    print("Heatmap 시각화")
    print()
    print("  파랑     : contact confidence 낮음")
    print("  초록/노랑: 중간")
    print("  빨강     : contact confidence 높음")
    print("  검은 점  : 직접 선택한 seed")
    print()
    print("  Q : 종료")
    print()

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Contact Heatmap Result",
        width=1280,
        height=900,
        mesh_show_back_face=True
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # validation
    # --------------------------------------------------------

    mesh_path = (
        Path(args.mesh)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(args.out_dir)
        .expanduser()
        .resolve()
    )

    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh 파일이 없습니다:\n"
            f"{mesh_path}"
        )

    if args.sigma_ratio <= 0:
        raise ValueError(
            "--sigma-ratio는 0보다 커야 합니다."
        )

    if args.cutoff_sigma < 0:
        raise ValueError(
            "--cutoff-sigma는 0 이상이어야 합니다."
        )

    if not (
        0.0 <=
        args.binary_threshold <=
        1.0
    ):
        raise ValueError(
            "--binary-threshold는 "
            "0~1 범위여야 합니다."
        )

    if args.point_size <= 0:
        raise ValueError(
            "--point-size는 0보다 커야 합니다."
        )

    # --------------------------------------------------------
    # mesh
    # --------------------------------------------------------

    mesh = load_mesh(
        mesh_path
    )

    vertices = np.asarray(
        mesh.vertices
    )

    triangles = np.asarray(
        mesh.triangles
    )

    # --------------------------------------------------------
    # annotation
    # --------------------------------------------------------

    seed_indices = (
        pick_contact_vertices(
            mesh,
            args.point_size
        )
    )

    # --------------------------------------------------------
    # heatmap
    # --------------------------------------------------------

    (
        H,
        geodesic_distance,
        sigma,
        bbox_diagonal
    ) = create_contact_heatmap(

        vertices,
        triangles,
        seed_indices,

        args.sigma_ratio,
        args.cutoff_sigma
    )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    (
        colored_mesh,
        npz_path,
        json_path,
        colored_path,
        seed_path
    ) = save_results(

        mesh,
        mesh_path,
        output_dir,

        seed_indices,

        H,
        geodesic_distance,

        sigma,
        bbox_diagonal,

        args.binary_threshold
    )

    # --------------------------------------------------------
    # log
    # --------------------------------------------------------

    print_statistics(
        H,
        seed_indices,
        sigma,
        bbox_diagonal
    )

    print()
    print("[INFO] 저장 완료")
    print()
    print(f"NPZ  : {npz_path}")
    print(f"JSON : {json_path}")
    print(f"PLY  : {colored_path}")
    print(f"Seed : {seed_path}")

    # --------------------------------------------------------
    # visualization
    # --------------------------------------------------------

    visualize_result(
        colored_mesh,
        vertices,
        seed_indices,
        bbox_diagonal
    )


if __name__ == "__main__":
    main()
