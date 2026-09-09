"""Loaders mapping do-as-i-do reconstruction outputs onto C2Dex symbols.

C2Dex Sec. III-A(b) assumes three inputs: an initial per-frame hand mesh
M_h,0_t = (V_h,0_t, F_h), a canonical object mesh M_o with a per-frame 6D pose
(R_o,t, p_o,t), and "the estimated camera" under which both are rendered.
The do-as-i-do pipeline produces all three, but from different modules:

    paper symbol            do-as-i-do source
    --------------------    ------------------------------------------------
    M_h,0_t                 {seq}/{task}/all_hand_meshes.npz  (HaWoR, MANO)
    M_o                     {seq}/video_segmentation/masks/
                              frame_{ref:06d}_masks/{obj}/{obj}.obj  (SAM 3D)
    (R_o,t, p_o,t)          {seq}/obj_tracking_out/{obj}/combined_visualization/
                              layout_camera_frame_optimized.json
    camera                  see `load_intrinsics` -- NOT a single camera

Both hand and object are already expressed in the camera frame: run_pipeline.sh
invokes HaWoR with --static_camera, which bypasses DROID-SLAM and fixes
R_c2w = I, t_c2w = 0, so HaWoR's "world" frame is the camera frame. The layout
JSON states its own convention in its `note` field: x-right, y-down, z-forward
(OpenCV).

Camera caveat
-------------
The paper renders both meshes under one camera. do-as-i-do has no such single
camera. MoGe re-estimates intrinsics independently on every frame, so the same
physical camera is assigned focal lengths spanning 854.6..964.0 px (mean 915.0,
std 22.6) across an 81-frame sequence -- estimation noise, not a real change.
HaWoR fixed one value for the whole sequence (the reference frame's, 915.125,
which happens to sit at the mean); track_object.py used each frame's own value,
so the object's translation_camera_frame was fit jointly with that per-frame
focal. Since u = f * X/Z + cx couples f and Z, re-projecting the object under a
different focal breaks the pairing it was fit under. See `load_intrinsics`.
"""

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# Layout JSON candidates, most-preferred first. The pipeline symlinks
# `layout_camera_frame_optimized.json` at the smoothed file (see run.txt);
# a broken symlink fails os.path.exists and falls through to the real files.
LAYOUT_CANDIDATES = (
    "layout_camera_frame_optimized.json",
    "layout_camera_frame_optimized_smooth.json",
    "layout_camera_frame_optimized_raw.json",
    "layout_camera_frame.json",
)


# HaWoR writes cam_space/{idx}/ per hand slot, not per tracklet:
# hawor_video.py's infiller stage uses `idx2hand = ['left', 'right']` with
# `tid = [0, 1]`.
HAND_TRACK_INDEX = {"left": 0, "right": 1}


@dataclass
class HandTrajectory:
    """M_h,0_t = (V_h,0_t, F_h), camera frame."""

    vertices: np.ndarray  # (T, V_h, 3) float64
    faces: np.ndarray  # (F_h, 3) int32
    valid: np.ndarray  # (T,) bool -- see `valid_source`
    side: str
    detected: np.ndarray  # (T,) bool -- frames the tracker actually detected
    infiller_valid: np.ndarray  # (T,) bool -- what all_hand_meshes.npz reports
    valid_source: str


@dataclass
class ObjectTrajectory:
    """M_o plus its per-frame pose (R_o,t, p_o,t), camera frame."""

    canonical_vertices: np.ndarray  # (V_o, 3) float64, already * mesh_scale
    faces: np.ndarray  # (F_o, 3) int32
    rotation: np.ndarray  # (T, 3, 3) float64
    translation: np.ndarray  # (T, 3) float64
    valid: np.ndarray  # (T,) bool -- pose present and in front of the camera
    mesh_scale: float
    name: str

    def posed_vertices(self, t: int) -> np.ndarray:
        """V_o,t = R_o,t @ (V_o * s) + p_o,t."""
        return self.canonical_vertices @ self.rotation[t].T + self.translation[t]


@dataclass
class Sequence:
    """Everything the silhouette stage needs for one reconstruction directory."""

    seq_dir: str
    task: str
    n_frames: int
    width: int
    height: int
    hand: HandTrajectory
    obj: ObjectTrajectory
    K_hand: np.ndarray  # (T, 3, 3)
    K_obj: np.ndarray  # (T, 3, 3)
    intrinsics_mode: str

    @property
    def frame_valid(self) -> np.ndarray:
        return self.hand.valid & self.obj.valid


def _read_config(seq_dir: str) -> dict:
    path = os.path.join(seq_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.json not found at {path} -- seq_dir must be a reconstruction "
            "pipeline output directory (the video's directory)."
        )
    return json.load(open(path))


def _frame_size(seq_dir: str) -> tuple[int, int]:
    """(width, height) read from the first extracted frame."""
    frames = sorted(glob.glob(os.path.join(seq_dir, "all_frames", "*.png")))
    if not frames:
        raise FileNotFoundError(
            f"No frames under {seq_dir}/all_frames -- run the reconstruction "
            "pipeline's frame-extraction step first."
        )
    import cv2

    img = cv2.imread(frames[0])
    if img is None:
        raise RuntimeError(f"Cannot read frame {frames[0]}")
    return img.shape[1], img.shape[0]


def load_track_coverage(seq_dir: str, side: str, n_frames: int) -> np.ndarray:
    """Frames where HaWoR's tracker actually detected the hand.

    HaWoR runs its hand model only on detected tracklet chunks and dumps each
    chunk to cam_space/{hand_idx}/{first}_{last}.json (inclusive frame range).
    Frames outside every chunk carry no observation at all: hawor_infiller
    hallucinates them with a transformer, and -- critically -- marks its own
    output valid, so `{side}_valid` in all_hand_meshes.npz does NOT flag them.

    On cupmove the left hand is covered by 0-13 and 22-76 only; frames 14-21
    (hand leaves the top of the image while transiting to the handle) and 77-80
    (hand withdraws out of frame) are pure infiller output.

    Returns (n_frames,) bool. Returns all-True with a warning if cam_space is
    absent, so callers degrade to the previous behaviour rather than failing.
    """
    task = os.path.basename(os.path.normpath(seq_dir))
    idx = HAND_TRACK_INDEX[side]
    track_dir = os.path.join(seq_dir, task, "cam_space", str(idx))
    if not os.path.isdir(track_dir):
        matches = sorted(glob.glob(os.path.join(seq_dir, "*", "cam_space", str(idx))))
        track_dir = matches[0] if matches else None

    covered = np.zeros(n_frames, dtype=bool)
    chunks = sorted(glob.glob(os.path.join(track_dir, "*.json"))) if track_dir else []
    if not chunks:
        print(
            f"  [warn] no cam_space/{idx} chunks for the {side} hand under {seq_dir}; "
            "cannot separate tracked frames from infilled ones."
        )
        return np.ones(n_frames, dtype=bool)

    for path in chunks:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            first, last = (int(x) for x in stem.split("_"))
        except ValueError:
            print(f"  [warn] unexpected cam_space chunk name {stem!r}, skipped")
            continue
        covered[max(first, 0):min(last + 1, n_frames)] = True
    return covered


def load_hand(
    seq_dir: str, side: str | None = None, valid_source: str = "infiller"
) -> HandTrajectory:
    """Load M_h,0_t from HaWoR's all_hand_meshes.npz (already camera frame).

    valid_source controls which frames downstream stages treat as usable:
        "infiller"  {side}_valid straight from all_hand_meshes.npz. Counts
                    infilled frames as valid -- the historical behaviour, and
                    what retargeting/pipeline/process_dataset.py uses.
        "tracks"    cam_space chunk coverage: only frames the tracker actually
                    detected.
        "both"      the conjunction (recommended).
    """
    if valid_source not in ("infiller", "tracks", "both"):
        raise ValueError(f"Unknown valid_source {valid_source!r}")

    task = os.path.basename(os.path.normpath(seq_dir))
    npz_path = os.path.join(seq_dir, task, "all_hand_meshes.npz")
    if not os.path.exists(npz_path):
        candidates = sorted(glob.glob(os.path.join(seq_dir, "*", "all_hand_meshes.npz")))
        if not candidates:
            raise FileNotFoundError(
                f"all_hand_meshes.npz not found under {seq_dir} -- it is written by "
                "the reconstruction pipeline's HaWoR step."
            )
        npz_path = candidates[0]

    if side is None:
        side = _read_config(seq_dir).get("anchor_hand", "right")

    d = np.load(npz_path)
    vertices = d[f"{side}_vertices"].astype(np.float64)
    if vertices.size == 0:
        raise ValueError(
            f"{npz_path} has no '{side}' hand data (shape {vertices.shape}); "
            f"config.json anchor_hand may disagree with the HaWoR output."
        )

    infiller_valid = d[f"{side}_valid"].astype(bool)
    detected = load_track_coverage(seq_dir, side, len(vertices))
    valid = {
        "infiller": infiller_valid,
        "tracks": detected,
        "both": infiller_valid & detected,
    }[valid_source]

    return HandTrajectory(
        vertices=vertices,
        faces=d[f"{side}_faces"].astype(np.int32),
        valid=valid,
        side=side,
        detected=detected,
        infiller_valid=infiller_valid,
        valid_source=valid_source,
    )


def resolve_layout_path(seq_dir: str, object_name: str) -> str:
    base = os.path.join(
        seq_dir, "obj_tracking_out", object_name, "combined_visualization"
    )
    for name in LAYOUT_CANDIDATES:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No usable layout JSON in {base} (tried {', '.join(LAYOUT_CANDIDATES)}). "
        "Note that a dangling symlink is skipped, not followed."
    )


def load_object(
    seq_dir: str, n_frames: int, object_name: str | None = None
) -> tuple[ObjectTrajectory, dict]:
    """Load M_o and its per-frame camera-frame pose (R_o,t, p_o,t)."""
    cfg = _read_config(seq_dir)
    if object_name is None:
        object_name = cfg["object_names"][0]

    layout_path = resolve_layout_path(seq_dir, object_name)
    layout = json.load(open(layout_path))

    # mesh_scale, not mesh_scale_original: the latter is SAM 3D's raw diffusion
    # scale, the former the pipeline's optimized one that translation_camera_frame
    # was solved against (matches retargeting/pipeline/process_dataset.py).
    mesh_scale = layout["translation_scale_optimization"]["mesh_scale"]

    mesh_glob = os.path.join(
        seq_dir,
        "video_segmentation",
        "masks",
        "frame_*_masks",
        object_name,
        f"{object_name}.obj",
    )
    mesh_candidates = sorted(glob.glob(mesh_glob))
    if not mesh_candidates:
        raise FileNotFoundError(f"Cannot find canonical object mesh: {mesh_glob}")
    mesh = trimesh.load(mesh_candidates[0], process=False, force="mesh")

    rotation = np.tile(np.eye(3), (n_frames, 1, 1))
    translation = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)
    for obj in layout["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if fi is None or not (0 <= fi < n_frames):
            continue
        ls = obj["local_to_scene"]
        t = np.asarray(ls["translation_camera_frame"], dtype=np.float64)
        q_wxyz = np.asarray(ls["quat_wxyz_camera_frame"], dtype=np.float64)
        if t[2] <= 0:  # behind the camera -- same guard as process_dataset.py
            continue
        rotation[fi] = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
        translation[fi] = t
        valid[fi] = True

    traj = ObjectTrajectory(
        canonical_vertices=np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        rotation=rotation,
        translation=translation,
        valid=valid,
        mesh_scale=mesh_scale,
        name=object_name,
    )
    return traj, layout


def _hand_focal(seq_dir: str) -> float:
    """The single focal HaWoR fit the whole sequence under."""
    task = os.path.basename(os.path.normpath(seq_dir))
    path = os.path.join(seq_dir, task, "img_focal.txt")
    if not os.path.exists(path):
        candidates = sorted(glob.glob(os.path.join(seq_dir, "*", "img_focal.txt")))
        if not candidates:
            raise FileNotFoundError(
                f"img_focal.txt not found under {seq_dir} -- it records the focal "
                "length HaWoR reconstructed the hand under."
            )
        path = candidates[0]
    return float(open(path).read().split()[0])


def _moge_focals(seq_dir: str, n_frames: int, layout: dict, width: int, height: int):
    """Per-frame MoGe focals, as used by track_object.py to fit the object pose.

    Primary source is all_frames/{idx:06d}_intrinsics.npy. Frames missing that
    file fall back to the layout entry's own `intrinsics_normalized`, which
    track_object.py wrote from the same matrix (fx = fx_norm * W, fy = fy_norm * H).
    """
    fx = np.full(n_frames, np.nan)
    fy = np.full(n_frames, np.nan)
    for fi in range(n_frames):
        path = os.path.join(seq_dir, "all_frames", f"{fi:06d}_intrinsics.npy")
        if os.path.exists(path):
            K = np.load(path)
            fx[fi], fy[fi] = K[0, 0], K[1, 1]

    for obj in layout["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        intr = obj.get("intrinsics_normalized")
        if intr is None or fi is None or not (0 <= fi < n_frames):
            continue
        if np.isnan(fx[fi]):
            fx[fi] = intr["fx_norm"] * width
            fy[fi] = intr["fy_norm"] * height

    if np.isnan(fx).any():
        fill = np.nanmean(fx), np.nanmean(fy)
        missing = np.where(np.isnan(fx))[0]
        print(
            f"  [warn] no per-frame intrinsics for frames {missing.tolist()}; "
            f"filling with the sequence mean ({fill[0]:.2f}, {fill[1]:.2f})"
        )
        fx[np.isnan(fx)], fy[np.isnan(fy)] = fill
    return fx, fy


def load_intrinsics(
    seq_dir: str,
    n_frames: int,
    layout: dict,
    width: int,
    height: int,
    mode: str = "per-source",
) -> tuple[np.ndarray, np.ndarray]:
    """Build (K_hand, K_obj), each (T, 3, 3).

    mode="per-source" (default)
        Each mesh is projected under the intrinsics it was actually fit with:
        the hand under HaWoR's fixed focal, the object under the per-frame MoGe
        focal that track_object.py solved its pose against. Both silhouettes
        then land where the hand and object really are in the image, which is
        what S_h,t and S_o,t mean in the paper. This departs from the paper's
        single-camera assumption -- do-as-i-do has no single camera to use.

    mode="unified"
        Both meshes under HaWoR's fixed focal, matching the paper's assumption
        literally. Measured cost on cupmove: the object silhouette moves by
        3.6 px on average (median 2.8, max 23.7) against a cup silhouette of
        ~98 px equivalent radius.

    Principal point is (W/2, H/2) in both modes: MoGe writes cx=W/2, cy=H/2 and
    the layout's cx_norm = cy_norm = 0.5 agrees.
    """
    if mode not in ("per-source", "unified"):
        raise ValueError(f"Unknown intrinsics mode {mode!r}")

    cx, cy = width / 2.0, height / 2.0
    f_hand = _hand_focal(seq_dir)

    K_hand = np.tile(np.eye(3), (n_frames, 1, 1))
    K_hand[:, 0, 0] = f_hand
    K_hand[:, 1, 1] = f_hand
    K_hand[:, 0, 2] = cx
    K_hand[:, 1, 2] = cy

    if mode == "unified":
        return K_hand, K_hand.copy()

    fx, fy = _moge_focals(seq_dir, n_frames, layout, width, height)
    K_obj = np.tile(np.eye(3), (n_frames, 1, 1))
    K_obj[:, 0, 0] = fx
    K_obj[:, 1, 1] = fy
    K_obj[:, 0, 2] = cx
    K_obj[:, 1, 2] = cy
    return K_hand, K_obj


def load_sequence(
    seq_dir: str,
    side: str | None = None,
    object_name: str | None = None,
    intrinsics_mode: str = "per-source",
    valid_source: str = "infiller",
) -> Sequence:
    seq_dir = os.path.abspath(seq_dir)
    task = os.path.basename(os.path.normpath(seq_dir))
    width, height = _frame_size(seq_dir)

    hand = load_hand(seq_dir, side, valid_source)
    n_frames = hand.vertices.shape[0]
    obj, layout = load_object(seq_dir, n_frames, object_name)
    K_hand, K_obj = load_intrinsics(
        seq_dir, n_frames, layout, width, height, intrinsics_mode
    )

    return Sequence(
        seq_dir=seq_dir,
        task=task,
        n_frames=n_frames,
        width=width,
        height=height,
        hand=hand,
        obj=obj,
        K_hand=K_hand,
        K_obj=K_obj,
        intrinsics_mode=intrinsics_mode,
    )


def gt_mask_path(seq_dir: str, frame_idx: int, name: str) -> str | None:
    """SAM2 segmentation mask for `name`, used only to validate the projection.

    The reference frame stores masks one directory deeper (alongside the meshes
    generated from them), every other frame stores them flat.
    """
    base = os.path.join(
        seq_dir, "video_segmentation", "masks", f"frame_{frame_idx:06d}_masks"
    )
    for candidate in (
        os.path.join(base, f"{name}.png"),
        os.path.join(base, name, f"{name}.png"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None
