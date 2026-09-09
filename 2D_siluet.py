import json
import numpy as np
import cv2


# ============================================================
# 설정
# ============================================================
FRAME_IDX = 20

# Cupmove 
# IMAGE_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/0020.png"
# HAND_NPZ_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/cupmove/all_hand_meshes.npz"
# OBJECT_OBJ_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/video_segmentation/masks/frame_000020_masks/cup/cup.obj"
# OBJECT_POSE_JSON_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/obj_tracking_out/cup/combined_visualization/layout_camera_frame_optimized_smooth.json"
# INTRINSICS_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/cupmove/0020_intrinsics.txt"

# Appleinput
IMAGE_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/appleinput/0046.png"
HAND_NPZ_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/appleinput/appleinput/all_hand_meshes.npz"
OBJECT_OBJ_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/appleinput/video_segmentation/masks/frame_000046_masks/apple/apple.obj"
OBJECT_POSE_JSON_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/appleinput/obj_tracking_out/apple/combined_visualization/layout_camera_frame_optimized_smooth.json"
INTRINSICS_PATH = "/home/intern/do-as-i-do/reconstruction/dataset/appleinput/0046_intrinsics.txt"

# 중요:
# cup.obj가 canonical SAM3D mesh라면 True일 가능성이 높음.
# 이미 scale이 적용된 mesh라면 False로 바꿔서 비교해보세요.
APPLY_JSON_SCALE = True

OUTPUT_PATH = "projection_overlay_0046.png"


# ============================================================
# Quaternion [w,x,y,z] -> rotation matrix
# ============================================================
def quat_wxyz_to_rotmat(q):
    w, x, y, z = q

    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")

    w, x, y, z = q / norm

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w)
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w)
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y)
        ]
    ], dtype=np.float64)


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
                values = line.split()
                vertices.append([
                    float(values[1]),
                    float(values[2]),
                    float(values[3])
                ])

            elif line.startswith("f "):
                values = line.split()[1:]

                # v/vt/vn 형태 대응
                idx = [
                    int(v.split("/")[0]) - 1
                    for v in values
                ]

                # triangle
                if len(idx) == 3:
                    faces.append(idx)

                # polygon이면 triangle fan으로 변환
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([
                            idx[0],
                            idx[i],
                            idx[i + 1]
                        ])

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32)
    )


# ============================================================
# 3D camera coordinate -> 2D pixel
# ============================================================
def project_points(vertices, fx, fy, cx, cy):
    X = vertices[:, 0]
    Y = vertices[:, 1]
    Z = vertices[:, 2]

    # camera 뒤의 점 방지
    valid = Z > 1e-6

    u = np.full(len(vertices), np.nan)
    v = np.full(len(vertices), np.nan)

    u[valid] = fx * X[valid] / Z[valid] + cx
    v[valid] = fy * Y[valid] / Z[valid] + cy

    return np.stack([u, v], axis=1), valid


# ============================================================
# mesh -> binary silhouette
# ============================================================
def rasterize_mesh(uv, vertices_3d, faces, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    for face in faces:
        # camera 뒤에 있는 triangle은 제외
        z = vertices_3d[face, 2]

        if np.any(z <= 1e-6):
            continue

        tri = uv[face]

        if np.any(~np.isfinite(tri)):
            continue

        pts = np.round(tri).astype(np.int32)

        cv2.fillConvexPoly(mask, pts, 255)

    return mask


# ============================================================
# 1. RGB image
# ============================================================
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

H, W = image.shape[:2]

print("Image size:", W, "x", H)


# ============================================================
# 2. Intrinsics
#
# txt:
# fx
# fy
# cx
# cy
# ============================================================
with open(INTRINSICS_PATH, "r") as f:
    intrinsic_values = [
        float(x.strip())
        for x in f.readlines()
        if x.strip()
    ]

fx, fy, cx, cy = intrinsic_values[:4]

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
])

print("\nK =")
print(K)


# ============================================================
# 3. MANO mesh
# ============================================================
hand_data = np.load(
    HAND_NPZ_PATH,
    allow_pickle=True
)

print("\nNPZ keys:")
print(hand_data.files)

hand_vertices = hand_data["left_vertices"][FRAME_IDX]

# faces가 frame별이 아니라 하나의 topology라고 가정
if "left_faces" in hand_data:
    hand_faces = hand_data["left_faces"]
else:
    raise KeyError("left_faces not found")

print("\nMANO vertices:", hand_vertices.shape)
print("MANO faces:", hand_faces.shape)

print("MANO xyz min:")
print(hand_vertices.min(axis=0))

print("MANO xyz max:")
print(hand_vertices.max(axis=0))

print("MANO centroid:")
print(hand_vertices.mean(axis=0))


# ============================================================
# 4. Object canonical mesh
# ============================================================
obj_vertices, obj_faces = load_obj(
    OBJECT_OBJ_PATH
)

print("\nObject vertices:", obj_vertices.shape)
print("Object faces:", obj_faces.shape)


# ============================================================
# 5. JSON camera-frame object pose
# ============================================================
with open(
    OBJECT_POSE_JSON_PATH,
    "r"
) as f:
    layout = json.load(f)


# frame_idx로 찾기
object_entry = None

for item in layout["objects"]:
    if item["frame_idx"] == FRAME_IDX:
        object_entry = item
        break

if object_entry is None:
    raise ValueError(
        f"frame {FRAME_IDX} not found"
    )


pose = object_entry["local_to_scene"]

translation = np.asarray(
    pose["translation_camera_frame"],
    dtype=np.float64
)

quat = np.asarray(
    pose["quat_wxyz_camera_frame"],
    dtype=np.float64
)

mesh_scale = layout["translation_scale_optimization"]["mesh_scale"]

scale = np.array(
    [mesh_scale, mesh_scale, mesh_scale],
    dtype=np.float64
)

R_obj = quat_wxyz_to_rotmat(quat)

print("\nObject camera translation:")
print(translation)

print("Object quaternion [w,x,y,z]:")
print(quat)

print("Object scale:")
print(scale)


# ============================================================
# 6. Object local mesh -> camera frame
# ============================================================

obj_local = obj_vertices.copy()

if APPLY_JSON_SCALE:
    obj_local = obj_local * scale


# column vector 기준으로는:
#
# X_cam = R X_local + t
#
# numpy Nx3 row-vector라서:
#
# X_cam = X_local @ R.T + t
#
obj_camera = (
    obj_local @ R_obj.T
    + translation
)


print("\nObject camera xyz min:")
print(obj_camera.min(axis=0))

print("Object camera xyz max:")
print(obj_camera.max(axis=0))

print("Object camera centroid:")
print(obj_camera.mean(axis=0))


# ============================================================
# 7. Projection
# ============================================================
hand_uv, _ = project_points(
    hand_vertices,
    fx, fy, cx, cy
)

obj_uv, _ = project_points(
    obj_camera,
    fx, fy, cx, cy
)


print("\nMANO 2D bbox:")
print(
    "u:",
    np.nanmin(hand_uv[:, 0]),
    "~",
    np.nanmax(hand_uv[:, 0])
)
print(
    "v:",
    np.nanmin(hand_uv[:, 1]),
    "~",
    np.nanmax(hand_uv[:, 1])
)


print("\nObject 2D bbox:")
print(
    "u:",
    np.nanmin(obj_uv[:, 0]),
    "~",
    np.nanmax(obj_uv[:, 0])
)
print(
    "v:",
    np.nanmin(obj_uv[:, 1]),
    "~",
    np.nanmax(obj_uv[:, 1])
)


# ============================================================
# 8. Mesh silhouettes
# ============================================================
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


# C2Dex candidate overlap
contact_mask = cv2.bitwise_and(
    hand_mask,
    obj_mask
)


# ============================================================
# 9. Overlay
#
# MANO     = green
# object   = red
# overlap  = yellow
# ============================================================
overlay = image.copy()


# Object silhouette: red
object_region = obj_mask > 0

red = np.zeros_like(image)
red[:, :, 2] = 255

overlay[object_region] = (
    0.55 * overlay[object_region]
    + 0.45 * red[object_region]
).astype(np.uint8)


# Hand silhouette: green
hand_region = hand_mask > 0

green = np.zeros_like(image)
green[:, :, 1] = 255

overlay[hand_region] = (
    0.45 * overlay[hand_region]
    + 0.55 * green[hand_region]
).astype(np.uint8)


# overlap: yellow
overlap_region = contact_mask > 0

yellow = np.zeros_like(image)
yellow[:, :, 1] = 255
yellow[:, :, 2] = 255

overlay[overlap_region] = (
    0.2 * overlay[overlap_region]
    + 0.8 * yellow[overlap_region]
).astype(np.uint8)


# ============================================================
# 10. contour 표시
# ============================================================
hand_contours, _ = cv2.findContours(
    hand_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

obj_contours, _ = cv2.findContours(
    obj_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# BGR
cv2.drawContours(
    overlay,
    obj_contours,
    -1,
    (0, 0, 255),
    2
)

cv2.drawContours(
    overlay,
    hand_contours,
    -1,
    (0, 255, 0),
    2
)


# ============================================================
# 11. 결과 저장
# ============================================================
cv2.imwrite(
    OUTPUT_PATH,
    overlay
)

print("\nSaved:")
print(OUTPUT_PATH)


# ============================================================
# 12. overlap 통계
# ============================================================
hand_area = np.count_nonzero(hand_mask)
obj_area = np.count_nonzero(obj_mask)
overlap_area = np.count_nonzero(contact_mask)

print("\nHand silhouette area :", hand_area)
print("Object silhouette area:", obj_area)
print("Overlap area          :", overlap_area)

if hand_area > 0:
    print(
        "Overlap / hand:",
        overlap_area / hand_area
    )


# ============================================================
# 화면 표시
# ============================================================
cv2.imshow(
    "Projection overlay",
    overlay
)

cv2.waitKey(0)
cv2.destroyAllWindows()
