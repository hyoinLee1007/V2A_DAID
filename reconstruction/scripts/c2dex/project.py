"""Pinhole projection and silhouette rasterization.

Camera convention is the one the object layout JSON declares in its `note`
field -- "camera frame (x-right, y-down, z-fwd)", i.e. OpenCV:

    u = fx * X/Z + cx,    v = fy * Y/Z + cy,    Z > 0

Silhouettes are the union of the projected triangles. C2Dex Sec. III-A(b)
defines S_h,t and S_o,t as image-space silhouettes, which is a 2D coverage
question, so no depth buffer is involved: a triangle occluded by another
triangle of the same mesh still lies inside that mesh's silhouette. Back-facing
triangles are likewise kept -- the paper filters hand vertices that occlude the
object without touching it later, via the normal-consistency score
w^n_t,i > gamma_n, not here.
"""

import cv2
import numpy as np


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixels.

    Args:
        points_cam: (N, 3) points in the camera frame.
        K: (3, 3) intrinsics.

    Returns:
        uv: (N, 2) float64 pixel coordinates. Rows with Z <= 0 are meaningless
            and flagged by `in_front`; they are not clamped or removed here so
            that the caller keeps a stable vertex indexing.
        in_front: (N,) bool, True where Z > 0.
    """
    z = points_cam[:, 2]
    in_front = z > 0
    safe_z = np.where(in_front, z, 1.0)
    uv = np.empty((len(points_cam), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * points_cam[:, 0] / safe_z + K[0, 2]
    uv[:, 1] = K[1, 1] * points_cam[:, 1] / safe_z + K[1, 2]
    uv[~in_front] = np.nan
    return uv, in_front


def rasterize_silhouette(
    points_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a mesh's image-space silhouette.

    Triangles with any vertex at or behind the camera plane are dropped: their
    projection is undefined and clipping them would not change the silhouette
    of a mesh that is otherwise in front of the camera.

    Returns:
        (height, width) bool mask.
    """
    uv, in_front = project_points(points_cam, K)
    mask = np.zeros((height, width), dtype=np.uint8)

    keep = in_front[faces].all(axis=1)
    if not keep.any():
        return mask.astype(bool)

    tris = np.round(uv[faces[keep]]).astype(np.int32)
    # cv2.fillPoly takes a sequence of polygons; an (n_tri, 3, 2) int32 array
    # fills every triangle in one call.
    cv2.fillPoly(mask, tris, 1)
    return mask.astype(bool)


def points_in_mask(uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Test which projected points land on a set pixel of `mask`.

    Points outside the image, or with NaN coordinates (Z <= 0), are False.
    """
    height, width = mask.shape
    u = np.rint(uv[:, 0])
    v = np.rint(uv[:, 1])
    finite = np.isfinite(u) & np.isfinite(v)
    ui = np.where(finite, u, -1).astype(np.int64)
    vi = np.where(finite, v, -1).astype(np.int64)
    inside = finite & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    out = np.zeros(len(uv), dtype=bool)
    out[inside] = mask[vi[inside], ui[inside]]
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0
