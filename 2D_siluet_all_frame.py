import os
import json
import glob
import cv2
import numpy as np


# ============================================================
# 경로 설정
# ============================================================

# 원본 RGB 이미지들이 들어있는 폴더
IMAGE_DIR = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/check_frames"
HAND_NPZ_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/cupmove/all_hand_meshes.npz"
OBJECT_OBJ_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/video_segmentation/masks/frame_000020_masks/cup/cup.obj"
OBJECT_POSE_JSON_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/obj_tracking_out/cup/combined_visualization/layout_camera_frame_optimized_smooth.json"
DEFAULT_INTRINSICS_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/0020_intrinsics.txt"


OUTPUT_DIR = "overlay_frames"
OUTPUT_VIDEO = "projection_overlay_all.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Quaternion [w,x,y,z] -> rotation matrix
# ============================================================

def quat_wxyz_to_rotmat(q):

    q = np.asarray(q, dtype=np.float64)

    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w)
        ],
        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w)
        ],
        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y)
        ]
    ])


# ============================================================
# OBJ loader
# ============================================================

def load_obj(path):

    vertices = []
    faces = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("v "):

                p = line.split()

                vertices.append([
                    float(p[1]),
                    float(p[2]),
                    float(p[3])
                ])

            elif line.startswith("f "):

                values = line.split()[1:]

                idx = [
                    int(v.split("/")[0]) - 1
                    for v in values
                ]

                if len(idx) == 3:

                    faces.append(idx)

                elif len(idx) > 3:

                    # polygon -> triangle fan
                    for i in range(1, len(idx)-1):

                        faces.append([
                            idx[0],
                            idx[i],
                            idx[i+1]
                        ])

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32)
    )


# ============================================================
# Camera projection
# ============================================================

def project_points(vertices, fx, fy, cx, cy):

    X = vertices[:, 0]
    Y = vertices[:, 1]
    Z = vertices[:, 2]

    uv = np.full(
        (len(vertices), 2),
        np.nan,
        dtype=np.float64
    )

    valid = Z > 1e-6

    uv[valid, 0] = \
        fx * X[valid] / Z[valid] + cx

    uv[valid, 1] = \
        fy * Y[valid] / Z[valid] + cy

    return uv


# ============================================================
# Mesh -> silhouette
# ============================================================

def rasterize_mesh(
    uv,
    vertices_3d,
    faces,
    width,
    height
):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for face in faces:

        # camera 뒤에 있는 triangle 제외
        if np.any(vertices_3d[face, 2] <= 1e-6):
            continue

        tri = uv[face]

        if not np.all(np.isfinite(tri)):
            continue

        pts = np.round(
            tri
        ).astype(np.int32)

        cv2.fillConvexPoly(
            mask,
            pts,
            255
        )

    return mask


# ============================================================
# Intrinsics loader
# ============================================================

def load_intrinsics(path):

    with open(path, "r") as f:

        values = [
            float(line.strip())
            for line in f.readlines()
            if line.strip()
        ]

    fx, fy, cx, cy = values[:4]

    return fx, fy, cx, cy


# ============================================================
# 데이터 로드
# ============================================================

hand_data = np.load(
    HAND_NPZ_PATH,
    allow_pickle=True
)

hand_vertices_all = \
    hand_data["left_vertices"]

hand_faces = \
    hand_data["left_faces"]


print(
    "MANO trajectory:",
    hand_vertices_all.shape
)


obj_vertices, obj_faces = \
    load_obj(OBJECT_OBJ_PATH)


with open(
    OBJECT_POSE_JSON_PATH,
    "r"
) as f:

    layout = json.load(f)


# ============================================================
# 중요:
# SAM3D 원래 scale(0.3938)이 아니라
# HaWoR/MoGe 정렬 이후의 최종 scale 사용
# ============================================================

mesh_scale = \
    layout[
        "translation_scale_optimization"
    ]["mesh_scale"]


print(
    "Final object mesh scale:",
    mesh_scale
)


# frame_idx -> object entry
object_dict = {

    item["frame_idx"]: item

    for item in layout["objects"]
}


# ============================================================
# RGB frame 검색
# ============================================================

image_paths = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.png"
        )
    )
)


print(
    "RGB frames:",
    len(image_paths)
)


num_hand = len(hand_vertices_all)
num_rgb = len(image_paths)
num_obj = len(object_dict)


print()
print("Number of frames")
print("----------------")
print("RGB    :", num_rgb)
print("MANO   :", num_hand)
print("Object :", num_obj)
print()


# 공통으로 처리 가능한 frame 수
T = min(
    num_rgb,
    num_hand,
    num_obj
)

print(
    "Processing frames:",
    T
)


# ============================================================
# intrinsics
# ============================================================

fx, fy, cx, cy = \
    load_intrinsics(
        DEFAULT_INTRINSICS_PATH
    )

print(
    "Intrinsics:",
    fx, fy, cx, cy
)


# ============================================================
# Video writer 준비
# ============================================================

first_image = cv2.imread(
    image_paths[0]
)

H, W = first_image.shape[:2]


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    10,
    (W, H)
)


# ============================================================
# 통계 저장
# ============================================================

stats = []


# ============================================================
# 전체 frame loop
# ============================================================

for t in range(T):

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    image = cv2.imread(
        image_paths[t]
    )

    if image is None:

        print(
            f"[Frame {t}] image load failed"
        )

        continue


    # --------------------------------------------------------
    # MANO camera-space mesh
    # --------------------------------------------------------

    hand_vertices = \
        hand_vertices_all[t]


    # --------------------------------------------------------
    # Object camera-space pose
    # --------------------------------------------------------

    if t not in object_dict:

        print(
            f"[Frame {t}] object pose missing"
        )

        continue


    entry = object_dict[t]

    pose = \
        entry["local_to_scene"]


    translation = np.asarray(
        pose[
            "translation_camera_frame"
        ],
        dtype=np.float64
    )


    quat = np.asarray(
        pose[
            "quat_wxyz_camera_frame"
        ],
        dtype=np.float64
    )


    R_obj = \
        quat_wxyz_to_rotmat(
            quat
        )


    # --------------------------------------------------------
    # Canonical cup mesh
    # -> final metric scale
    # -> rotation
    # -> camera translation
    # --------------------------------------------------------

    obj_local = \
        obj_vertices * mesh_scale


    obj_camera = \
        obj_local @ R_obj.T \
        + translation


    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------

    hand_uv = project_points(
        hand_vertices,
        fx, fy, cx, cy
    )

    obj_uv = project_points(
        obj_camera,
        fx, fy, cx, cy
    )


    # --------------------------------------------------------
    # Silhouette
    # --------------------------------------------------------

    hand_mask = rasterize_mesh(
        hand_uv,
        hand_vertices,
        hand_faces,
        W,
        H
    )


    obj_mask = rasterize_mesh(
        obj_uv,
        obj_camera,
        obj_faces,
        W,
        H
    )


    contact_mask = \
        cv2.bitwise_and(
            hand_mask,
            obj_mask
        )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = image.copy()


    # -------------------------
    # Object = RED
    # -------------------------

    object_region = \
        obj_mask > 0

    red = np.zeros_like(image)

    red[:, :, 2] = 255


    overlay[object_region] = (

        0.55 *
        overlay[object_region]

        +

        0.45 *
        red[object_region]

    ).astype(np.uint8)


    # -------------------------
    # Hand = GREEN
    # -------------------------

    hand_region = \
        hand_mask > 0

    green = np.zeros_like(image)

    green[:, :, 1] = 255


    overlay[hand_region] = (

        0.45 *
        overlay[hand_region]

        +

        0.55 *
        green[hand_region]

    ).astype(np.uint8)


    # -------------------------
    # overlap = YELLOW
    # -------------------------

    overlap_region = \
        contact_mask > 0

    yellow = np.zeros_like(image)

    yellow[:, :, 1] = 255
    yellow[:, :, 2] = 255


    overlay[overlap_region] = (

        0.15 *
        overlay[overlap_region]

        +

        0.85 *
        yellow[overlap_region]

    ).astype(np.uint8)


    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    hand_contours, _ = \
        cv2.findContours(
            hand_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


    obj_contours, _ = \
        cv2.findContours(
            obj_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


    cv2.drawContours(
        overlay,
        hand_contours,
        -1,
        (0, 255, 0),
        2
    )


    cv2.drawContours(
        overlay,
        obj_contours,
        -1,
        (0, 0, 255),
        2
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    hand_area = \
        np.count_nonzero(
            hand_mask
        )

    object_area = \
        np.count_nonzero(
            obj_mask
        )

    overlap_area = \
        np.count_nonzero(
            contact_mask
        )


    if hand_area > 0:

        overlap_ratio = \
            overlap_area / hand_area

    else:

        overlap_ratio = 0


    stats.append([
        t,
        hand_area,
        object_area,
        overlap_area,
        overlap_ratio
    ])


    # --------------------------------------------------------
    # Frame 번호 표시
    # --------------------------------------------------------

    cv2.putText(
        overlay,
        f"Frame {t:04d}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        overlay,
        f"overlap/hand = {overlap_ratio:.3f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"overlay_{t:04d}.png"
    )


    cv2.imwrite(
        output_path,
        overlay
    )


    writer.write(
        overlay
    )


    print(
        f"Frame {t:03d}"
        f" | hand={hand_area:6d}"
        f" | object={object_area:6d}"
        f" | overlap={overlap_area:6d}"
        f" | ratio={overlap_ratio:.4f}"
    )


writer.release()


# ============================================================
# CSV 저장
# ============================================================

stats = np.asarray(
    stats
)


np.savetxt(
    "projection_stats.csv",
    stats,
    delimiter=",",
    header=(
        "frame,"
        "hand_area,"
        "object_area,"
        "overlap_area,"
        "overlap_over_hand"
    ),
    comments=""
)


print()
print("Finished.")
print()
print("Overlay images:")
print(OUTPUT_DIR)
print()
print("Video:")
print(OUTPUT_VIDEO)
print()
print("Statistics:")
print("projection_stats.csv")
