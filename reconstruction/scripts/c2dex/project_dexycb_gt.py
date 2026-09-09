"""Project the DexYCB ground-truth MANO hand + object pose onto the raw RGB
frames, independent of the do-as-i-do pipeline. Reuses the projection/
rasterization logic from 2D_siluet.py at the repo root, adapted to loop over
multiple frames and use the master camera's own calibrated intrinsics
(extrinsics.yml identity camera) instead of MoGe's per-frame estimate --
this is a pure "is the GT good" check, so it should lean on DexYCB's own
calibration, not do-as-i-do's vision-estimated cameras.

Usage: python project_dexycb_gt.py
"""

import json
import os

import cv2
import numpy as np

REPO = "/home/intern/do-as-i-do"
FRAMES_DIR = f"{REPO}/reconstruction/dataset/dexycb/all_frames"
HAND_NPZ_PATH = f"{REPO}/reconstruction/dataset/dexycb_gt/dexycb_gt/all_hand_meshes.npz"
OBJECT_OBJ_PATH = "/home/intern/Downloads/models/019_pitcher_base/textured_simple.obj"
LAYOUT_JSON_PATH = (
    f"{REPO}/reconstruction/dataset/dexycb_gt/obj_tracking_out/pitcher/"
    "combined_visualization/layout_camera_frame_optimized.json"
)
OUT_DIR = f"{REPO}/reconstruction/dataset/dexycb_gt_overlay"

# Master camera (840412060917) calibrated color intrinsics, 640x480
# (calibration/intrinsics/840412060917_640x480.yml -- DexYCB's own factory
# RealSense calibration, not a vision estimate).
FX, FY, CX, CY = 621.7969970703125, 621.4028930664062, 302.4136962890625, 236.88433837890625

FRAMES = [0, 5, 10, 13, 20, 30, 40, 50, 60, 65, 70, 73]


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_obj(path):
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def project_points(vertices, fx, fy, cx, cy):
    X, Y, Z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    valid = Z > 1e-6
    u = np.full(len(vertices), np.nan)
    v = np.full(len(vertices), np.nan)
    u[valid] = fx * X[valid] / Z[valid] + cx
    v[valid] = fy * Y[valid] / Z[valid] + cy
    return np.stack([u, v], axis=1)


def rasterize_mesh(uv, vertices_3d, faces, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        if np.any(vertices_3d[face, 2] <= 1e-6):
            continue
        tri = uv[face]
        if np.any(~np.isfinite(tri)):
            continue
        cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255)
    return mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hand_data = np.load(HAND_NPZ_PATH)
    hand_vertices_all = hand_data["left_vertices"]
    hand_faces = hand_data["left_faces"]

    obj_local, obj_faces = load_obj(OBJECT_OBJ_PATH)
    layout = json.load(open(LAYOUT_JSON_PATH))
    by_frame = {o["frame_idx"]: o["local_to_scene"] for o in layout["objects"]}

    for fi in FRAMES:
        img_path = f"{FRAMES_DIR}/{fi:06d}.png"
        image = cv2.imread(img_path)
        if image is None:
            print(f"skip frame {fi}: image not found at {img_path}")
            continue
        H, W = image.shape[:2]

        hand_vertices = hand_vertices_all[fi]
        pose = by_frame[fi]
        t = np.asarray(pose["translation_camera_frame"])
        q = np.asarray(pose["quat_wxyz_camera_frame"])
        R = quat_wxyz_to_rotmat(q)
        obj_cam = obj_local @ R.T + t

        hand_uv = project_points(hand_vertices, FX, FY, CX, CY)
        obj_uv = project_points(obj_cam, FX, FY, CX, CY)

        hand_mask = rasterize_mesh(hand_uv, hand_vertices, hand_faces, W, H)
        obj_mask = rasterize_mesh(obj_uv, obj_cam, obj_faces, W, H)
        contact_mask = cv2.bitwise_and(hand_mask, obj_mask)

        overlay = image.copy()
        red, green, yellow = np.zeros_like(image), np.zeros_like(image), np.zeros_like(image)
        red[:, :, 2] = 255
        green[:, :, 1] = 255
        yellow[:, :, 1] = 255
        yellow[:, :, 2] = 255

        m = obj_mask > 0
        overlay[m] = (0.55 * overlay[m] + 0.45 * red[m]).astype(np.uint8)
        m = hand_mask > 0
        overlay[m] = (0.45 * overlay[m] + 0.55 * green[m]).astype(np.uint8)
        m = contact_mask > 0
        overlay[m] = (0.2 * overlay[m] + 0.8 * yellow[m]).astype(np.uint8)

        hc, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        oc, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, oc, -1, (0, 0, 255), 2)
        cv2.drawContours(overlay, hc, -1, (0, 255, 0), 2)
        cv2.putText(overlay, f"frame {fi}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = f"{OUT_DIR}/overlay_{fi:06d}.png"
        cv2.imwrite(out_path, overlay)
        print(f"frame {fi}: hand_area={np.count_nonzero(hand_mask)} obj_area={np.count_nonzero(obj_mask)} "
              f"overlap={np.count_nonzero(contact_mask)}  -> {out_path}")


if __name__ == "__main__":
    main()
