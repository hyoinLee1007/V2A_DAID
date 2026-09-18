#!/usr/bin/env python3
"""
Per-frame translation scale optimization via hand-anchored pointmap alignment.

The anchor hand mesh from HaWoR is treated as ground truth. The pointmap (MoGe)
gives the relative 3D offset between hand and object, but has its own global
scale. We compute the pointmap-to-real scale factor per frame by comparing the
anchor hand's depth in the pointmap vs HaWoR, then use the scaled hand-to-object
vector from the pointmap to place the object relative to the hand.

For each frame:
  1. h_real  = anchor hand centroid from HaWoR (ground truth)
  2. h_pm    = pointmap sampled at anchor hand projected pixels
  3. o_pm    = pointmap median at object mask pixels
  4. k       = h_real_z / h_pm_z  (pointmap-to-real depth scale)
  5. obj_pos = h_real + k * (o_pm - h_pm)
  6. Solve for translation_scale s: mesh_center(s) ≈ obj_pos

Usage (simplified — most paths inferred from --video-dir and --layout-json):
  python optimize_translation_scale.py \
    --video-dir /path/to/video_dir \
    --layout-json /path/to/video_dir/obj_tracking_out/bowl/combined_visualization/layout_camera_frame.json \
    --anchor-hand left \
    --ref-frame 0

All other arguments (--mesh, --scale, --pointmap-dir, --mask-dir, --mask-name,
--hand-meshes, --output, --viz-dir, --frames-dir) are inferred but can be
overridden explicitly.
"""

import argparse
import copy
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

MAX_HAND_RAYS = 2000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-frame translation scale optimization via hand-anchored pointmap alignment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Video directory root. When provided, most other paths are inferred automatically.')
    parser.add_argument('--layout-json', type=str, required=True,
                        help='Camera-frame layout JSON (with translation_camera_frame and quat_wxyz_camera_frame)')
    parser.add_argument('--mesh', type=str, default=None,
                        help='Object mesh (.obj)')
    parser.add_argument('--scale', type=float, default=None,
                        help='Mesh scale factor')
    parser.add_argument('--pointmap-dir', type=str, default=None,
                        help='Directory with per-frame pointmaps (NNNN_pointmap.npy) and intrinsics')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Directory with per-frame masks (frame_NNNNNN_masks/<mask-name>.png)')
    parser.add_argument('--mask-name', type=str, default=None,
                        help='Name of the object mask file (without .png)')
    parser.add_argument('--hand-meshes', type=str, default=None,
                        help='Hand meshes NPZ (already in camera frame, from HaWoR)')
    parser.add_argument('--anchor-hand', type=str, default='left',
                        choices=['left', 'right'],
                        help='Which hand to use as the anchor for pointmap scale calibration')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--ref-frame', type=int, default=None,
                        help='Reference frame: compute pointmap scale k on this frame first, then '
                             'rescale mesh and all translations by k before running per-frame optimization')
    parser.add_argument('--min-mask-pixels', type=int, default=100,
                        help='Skip frames with fewer mask pixels than this')
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='If set, save per-frame debug visualizations to this directory')
    parser.add_argument('--frames-dir', type=str, default=None,
                        help='Directory with frame images (NNNNNN.png) for visualization (required if --viz-dir is set)')
    args = parser.parse_args()
    resolve_args(args)
    return args


def resolve_args(args):
    """Infer missing arguments from --video-dir and --layout-json."""
    import re

    # Infer mask-name from layout-json path
    if args.mask_name is None:
        m = re.search(r'(?:tracking_output_every_frame|guided_pose_prediction|sweep_k2_50samples|obj_tracking_out)/([^/]+)/combined_visualization/',
                      args.layout_json)
        if m:
            args.mask_name = m.group(1)
            print(f"[inferred] --mask-name = {args.mask_name}")
        else:
            print("[error] Cannot infer --mask-name from layout-json path. "
                  "Expected '.../tracking_output_every_frame/<name>/combined_visualization/...' or "
                  "'.../guided_pose_prediction/<name>/combined_visualization/...'")
            sys.exit(1)

    # Infer scale from layout JSON
    if args.scale is None:
        with open(args.layout_json) as f:
            layout = json.load(f)
        args.scale = float(layout["objects"][0]["local_to_scene"]["scale"][0])
        print(f"[inferred] --scale = {args.scale}")

    video_dir = args.video_dir

    if video_dir is not None:
        video_dir_name = os.path.basename(os.path.normpath(video_dir))

        if args.pointmap_dir is None:
            args.pointmap_dir = os.path.join(video_dir, "all_frames")
            print(f"[inferred] --pointmap-dir = {args.pointmap_dir}")

        if args.mask_dir is None:
            args.mask_dir = os.path.join(video_dir, "video_segmentation", "masks")
            print(f"[inferred] --mask-dir = {args.mask_dir}")

        if args.hand_meshes is None:
            args.hand_meshes = os.path.join(video_dir, video_dir_name, "all_hand_meshes.npz")
            print(f"[inferred] --hand-meshes = {args.hand_meshes}")

        if args.mesh is None:
            if args.ref_frame is None:
                print("[error] Cannot infer --mesh without --ref-frame. Provide --mesh or --ref-frame.")
                sys.exit(1)
            args.mesh = os.path.join(video_dir, "video_segmentation", "masks",
                                     f"frame_{args.ref_frame:06d}_masks",
                                     args.mask_name, f"{args.mask_name}.obj")
            print(f"[inferred] --mesh = {args.mesh}")
            if not os.path.exists(args.mesh):
                print(f"[error] Mesh not found: {args.mesh}")
                sys.exit(1)

        # Auto-enable viz when --video-dir is provided
        if args.viz_dir is None:
            args.viz_dir = os.path.join(os.path.dirname(args.layout_json), "viz")
            print(f"[inferred] --viz-dir = {args.viz_dir}")
        if args.frames_dir is None:
            args.frames_dir = os.path.join(video_dir, "all_frames")
            print(f"[inferred] --frames-dir = {args.frames_dir}")

    if args.output is None:
        args.output = args.layout_json.replace(".json", "_optimized.json")
        print(f"[inferred] --output = {args.output}")

    # Final validation: all required values must be set
    required = {'mesh': args.mesh, 'scale': args.scale, 'pointmap_dir': args.pointmap_dir,
                'mask_dir': args.mask_dir, 'mask_name': args.mask_name,
                'hand_meshes': args.hand_meshes, 'output': args.output}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"[error] Missing required arguments (provide --video-dir or set explicitly): "
              f"{', '.join('--' + k.replace('_', '-') for k in missing)}")
        sys.exit(1)


def load_layout_camera_frame(json_path):
    """Load camera-frame layout JSON. Returns (data, frames_list)."""
    with open(json_path) as f:
        data = json.load(f)

    frames = []
    for i, obj in enumerate(data["objects"]):
        frame_idx = obj.get("frame_index", obj.get("frame_idx"))
        if frame_idx is None:
            continue
        pose = obj["local_to_scene"]
        if "translation_camera_frame" not in pose or "quat_wxyz_camera_frame" not in pose:
            print(f"[warn] frame {frame_idx}: missing camera-frame fields, skipping")
            continue
        frames.append({
            "obj_index": i,
            "frame_idx": frame_idx,
            "t_cam": np.array(pose["translation_camera_frame"]),
            "quat_wxyz_cam": pose["quat_wxyz_camera_frame"],
        })

    frames.sort(key=lambda x: x["frame_idx"])
    return data, frames


def get_rot_matrix(quat_wxyz):
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def load_pointmap_and_intrinsics(pointmap_dir, frame_idx):
    """Load pointmap and intrinsics for a given frame index."""
    pm_path = os.path.join(pointmap_dir, f"{frame_idx:06d}_pointmap.npy")
    intr_path = os.path.join(pointmap_dir, f"{frame_idx:06d}_intrinsics.npy")
    if not os.path.exists(pm_path) or not os.path.exists(intr_path):
        return None, None
    return np.load(pm_path), np.load(intr_path)


def load_mask(mask_dir, frame_idx, mask_name):
    """Load binary object mask for a given frame index."""
    path = os.path.join(mask_dir, f"frame_{frame_idx:06d}_masks", f"{mask_name}.png")
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask > 127


def project_to_pixels(verts_cam, fx, fy, cx, cy, w, h):
    """Project 3D camera-frame vertices to pixel coordinates, clipped to image bounds."""
    u = (fx * verts_cam[:, 0] / verts_cam[:, 2] + cx).astype(int).clip(0, w - 1)
    v = (fy * verts_cam[:, 1] / verts_cam[:, 2] + cy).astype(int).clip(0, h - 1)
    return u, v


def raycast_first_hits(verts, faces, mask, fx, fy, cx, cy, max_rays=MAX_HAND_RAYS, rng=None):
    """Shoot a ray from the camera origin through each True pixel in `mask` and return the
    first intersection on the mesh (verts, faces). Rays that miss are dropped.

    Used to recover *front-surface* samples: averaging hits gives a centroid that
    excludes back-facing geometry, which a 2D silhouette filter cannot do.

    Args:
        verts: (V, 3) camera-frame vertices.
        faces: (F, 3) face indices.
        mask:  (H, W) bool mask of pixels to raycast from.
        fx, fy, cx, cy: pinhole intrinsics matching mask's pixel grid.
        max_rays: subsample mask pixels down to this many rays (uniform random).
        rng: optional np.random.Generator for the subsample.

    Returns:
        hits:  (M, 3) first-hit 3D points in camera frame.
        hit_u: (M,) pixel u of rays that hit.
        hit_v: (M,) pixel v of rays that hit.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0, dtype=int)
    if max_rays is not None and len(xs) > max_rays:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(xs), size=max_rays, replace=False)
        xs, ys = xs[idx], ys[idx]
    dirs = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.zeros_like(dirs)
    tm = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                         faces=np.asarray(faces), process=False)
    locations, index_ray, _ = tm.ray.intersects_location(origins, dirs, multiple_hits=False)
    return locations, xs[index_ray], ys[index_ray]


def compute_optimal_scale(c_rot, t_cam, target_3d):
    """
    Find translation_scale s that minimizes ||c_rot + t_cam * s - target||^2.

    s* = t_cam . (target - c_rot) / (t_cam . t_cam)
    """
    residual = target_3d - c_rot
    s = np.dot(t_cam, residual) / np.dot(t_cam, t_cam)
    return s


def main():
    args = parse_args()

    # Validate paths
    for path, name in [(args.layout_json, "Layout JSON"), (args.mesh, "Mesh"),
                       (args.pointmap_dir, "Pointmap dir"), (args.mask_dir, "Mask dir"),
                       (args.hand_meshes, "Hand meshes")]:
        if not os.path.exists(path):
            print(f"[error] {name} not found: {path}")
            sys.exit(1)

    # Load layout
    print("Loading layout JSON...")
    layout_data, frames = load_layout_camera_frame(args.layout_json)
    print(f"  {len(frames)} frames with camera-frame poses")

    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load_mesh(args.mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh_verts = np.array(mesh.vertices, dtype=np.float64)
    print(f"  {len(mesh_verts)} vertices, {len(mesh.faces)} faces")

    # Load hand meshes
    print(f"Loading hand meshes (anchor: {args.anchor_hand})...")
    hand_data = np.load(args.hand_meshes)
    anchor = args.anchor_hand
    n_hand_frames = hand_data[f'{anchor}_vertices'].shape[0]
    hand_faces = np.asarray(hand_data[f'{anchor}_faces'])
    print(f"  {anchor} hand: {n_hand_frames} frames, {hand_data[f'{anchor}_vertices'].shape[1]} vertices, "
          f"{len(hand_faces)} faces")

    rng = np.random.default_rng(0)

    # Visualization setup
    viz_dir = args.viz_dir
    if viz_dir:
        if not args.frames_dir:
            print("[error] --frames-dir is required when --viz-dir is set")
            sys.exit(1)
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Saving visualizations to {viz_dir}")

    # Reference frame: compute pointmap scale k and use it to rescale mesh + translations
    mesh_scale = args.scale
    ref_k = None
    if args.ref_frame is not None:
        ref_fr = None
        for fr in frames:
            if fr["frame_idx"] == args.ref_frame:
                ref_fr = fr
                break
        if ref_fr is None:
            print(f"[error] --ref-frame {args.ref_frame} not found in layout")
            sys.exit(1)

        print(f"\nComputing pointmap scale k from ref frame {args.ref_frame}...")
        fidx = ref_fr["frame_idx"]
        pointmap, intrinsics = load_pointmap_and_intrinsics(args.pointmap_dir, fidx)
        if pointmap is None:
            print(f"[error] pointmap not found for ref frame {fidx}")
            sys.exit(1)
        pm_h, pm_w = pointmap.shape[:2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        hi = min(fidx, n_hand_frames - 1)
        anchor_verts_ref = hand_data[f'{anchor}_vertices'][hi]
        hand_mask = load_mask(args.mask_dir, fidx, f"{anchor}_hand_0")
        if hand_mask is None or not hand_mask.any():
            # SAM3 writes an all-zero mask on frames where it lost the object,
            # so an existing file is not evidence of a hand. Without this the
            # failure surfaces two steps later as "too few hand raycast hits (0)".
            why = "not found" if hand_mask is None else "empty (0 px)"
            print(f"[error] ref frame {fidx}: {anchor}_hand_0 mask is {why}")
            usable = []
            for fr in frames:
                m = load_mask(args.mask_dir, fr["frame_idx"], f"{anchor}_hand_0")
                if m is not None and m.any():
                    usable.append(fr["frame_idx"])
            if usable:
                near = sorted(usable, key=lambda f: abs(f - fidx))[:3]
                print(f"        frames with a non-empty hand mask nearest to it: {near}"
                      f" — pass one with --ref-frame")
            sys.exit(1)
        if hand_mask.shape != (pm_h, pm_w):
            hand_mask = cv2.resize(hand_mask.astype(np.uint8), (pm_w, pm_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        # Front-surface centroid via raycasting through hand-mask pixels: h_real and h_pm
        # are averaged over the same set of pixels (rays that hit the HaWoR mesh), so the
        # depth ratio compares the same physical region without back-facing vertex bias.
        hits_ref, hu_ref, hv_ref = raycast_first_hits(
            anchor_verts_ref, hand_faces, hand_mask, fx, fy, cx, cy,
            max_rays=MAX_HAND_RAYS, rng=rng,
        )
        if len(hits_ref) < 10:
            print(f"[error] ref frame: too few hand raycast hits ({len(hits_ref)})")
            sys.exit(1)
        h_real = hits_ref.mean(axis=0)
        h_pm = pointmap[hv_ref, hu_ref].mean(axis=0)

        if abs(h_pm[2]) < 1e-6:
            print(f"[error] ref frame hand pointmap depth ~0")
            sys.exit(1)
        ref_k = float(h_real[2] / h_pm[2])

        # Save ref frame intrinsics for hand projection across all frames
        hand_fx, hand_fy = fx, fy
        hand_cx, hand_cy = cx, cy
        hand_proj_w, hand_proj_h = pm_w, pm_h
        print(f"  using ref frame intrinsics for hand projection: fx={hand_fx:.1f} fy={hand_fy:.1f} cx={hand_cx:.1f} cy={hand_cy:.1f}")

        mesh_scale = args.scale * ref_k
        print(f"  ref k: {ref_k:.6f}")
        print(f"  new mesh_scale: {args.scale} * {ref_k:.6f} = {mesh_scale:.6f}")

        # Scale all translations by ref_k
        for fr in frames:
            fr["t_cam_orig"] = fr["t_cam"].copy()
            fr["t_cam"] = fr["t_cam"] * ref_k
        print(f"  scaled all t_cam by {ref_k:.6f}")

    # Process each frame
    print(f"\nOptimizing translation_scale per frame (mesh_scale={mesh_scale:.6f})...")
    output_data = copy.deepcopy(layout_data)
    per_frame_scales = []
    skipped = 0

    for i, fr in enumerate(frames):
        fidx = fr["frame_idx"]
        rot_matrix = get_rot_matrix(fr["quat_wxyz_cam"])
        t_cam = fr["t_cam"]

        # Load pointmap and intrinsics
        pointmap, intrinsics = load_pointmap_and_intrinsics(args.pointmap_dir, fidx)
        if pointmap is None:
            print(f"  frame {fidx:3d}: pointmap/intrinsics not found, skipping")
            skipped += 1
            continue

        # Load object mask
        mask = load_mask(args.mask_dir, fidx, args.mask_name)
        if mask is None:
            print(f"  frame {fidx:3d}: mask not found, skipping")
            skipped += 1
            continue

        # Resize mask if needed
        pm_h, pm_w = pointmap.shape[:2]
        if mask.shape != (pm_h, pm_w):
            mask = cv2.resize(mask.astype(np.uint8), (pm_w, pm_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        n_pixels = mask.sum()
        if n_pixels < args.min_mask_pixels: # TODO: handle this case better
            print(f"  frame {fidx:3d}: mask too small ({n_pixels} px), skipping")
            skipped += 1
            continue

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Intrinsics for hand projection: ref frame intrinsics if available, else per-frame
        hfx = hand_fx if ref_k is not None else fx
        hfy = hand_fy if ref_k is not None else fy
        hcx = hand_cx if ref_k is not None else cx
        hcy = hand_cy if ref_k is not None else cy

        if i == 0:
            print(f"  hand proj intrinsics: fx={hfx:.1f} fy={hfy:.1f} cx={hcx:.1f} cy={hcy:.1f}")
            print(f"  frame pointmap intr:  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        # Anchor hand vertices from HaWoR (this frame's pose)
        hi = min(fidx, n_hand_frames - 1)
        anchor_verts = hand_data[f'{anchor}_vertices'][hi]

        # Hand mask, resized to pointmap resolution
        hand_mask_name = f"{anchor}_hand_0"
        hand_mask = load_mask(args.mask_dir, fidx, hand_mask_name)
        if hand_mask is None or not hand_mask.any():
            # The translation written for this frame is the raw tracker one,
            # not a ref-scaled one; process_dataset.py treats frames absent
            # from per_frame as invalid and interpolates them.
            why = "missing" if hand_mask is None else "empty"
            print(f"  frame {fidx:3d}: hand mask {why}, skipping (not in per_frame; interpolated downstream)")
            skipped += 1
            continue
        if hand_mask.shape != (pm_h, pm_w):
            hand_mask = cv2.resize(hand_mask.astype(np.uint8), (pm_w, pm_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        # Front-surface centroid via raycasting: each hand-mask pixel shoots a ray from the
        # camera and we keep the first hit on the HaWoR mesh. h_real and h_pm are averaged
        # over the same hit-pixel set, so k = h_real_z / h_pm_z compares the same physical
        # region front-surface-only (no back-facing-vertex bias).
        hand_hits, hand_hit_u, hand_hit_v = raycast_first_hits(
            anchor_verts, hand_faces, hand_mask, hfx, hfy, hcx, hcy,
            max_rays=MAX_HAND_RAYS, rng=rng,
        )
        if len(hand_hits) < 10:
            print(f"  frame {fidx:3d}: too few hand raycast hits ({len(hand_hits)}), skipping")
            skipped += 1
            continue
        h_real = hand_hits.mean(axis=0)
        h_pm = pointmap[hand_hit_v, hand_hit_u].mean(axis=0)

        # Object centroid from pointmap
        o_pm = np.mean(pointmap[mask], axis=0)

        # Pointmap-to-real scale factor (from hand depth comparison)
        if abs(h_pm[2]) < 1e-6:
            print(f"  frame {fidx:3d}: hand pointmap depth ~0, skipping")
            skipped += 1
            continue
        k = h_real[2] / h_pm[2]

        # Object target position: hand anchor + scaled relative offset
        obj_target = h_real + k * (o_pm - h_pm)

        # Compute visible-surface centroid of the rotated mesh (filtered by object mask)
        verts_rotated = (mesh_verts * mesh_scale) @ rot_matrix.T
        verts_with_t = verts_rotated + t_cam
        mesh_u, mesh_v = project_to_pixels(verts_with_t, fx, fy, cx, cy, pm_w, pm_h)
        mesh_in_mask = mask[mesh_v, mesh_u]
        if mesh_in_mask.sum() < 10:
            print(f"  frame {fidx:3d}: too few mesh pixels in object mask ({mesh_in_mask.sum()}), skipping")
            skipped += 1
            continue
        c_rot = verts_rotated[mesh_in_mask].mean(axis=0)

        # Solve for translation_scale
        opt_scale = compute_optimal_scale(c_rot, t_cam, obj_target)

        # Errors before/after
        pos_before = c_rot + t_cam * 1.0
        pos_after = c_rot + t_cam * opt_scale
        err_before = np.linalg.norm(pos_before - obj_target)
        err_after = np.linalg.norm(pos_after - obj_target)

        print(f"  frame {fidx:3d}  [{i+1}/{len(frames)}]  "
              f"scale: {opt_scale:.4f}  pm_k: {k:.4f}  "
              f"err: {err_before:.4f} -> {err_after:.4f}")

        # Visualization
        if viz_dir:
            img_path = os.path.join(args.frames_dir, f"{fidx:06d}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(args.frames_dir, f"{fidx:06d}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                vis = img.copy()

                # Object mask overlay (green)
                mask_full = load_mask(args.mask_dir, fidx, args.mask_name)
                if mask_full is not None:
                    if mask_full.shape != (img_h, img_w):
                        mask_full = cv2.resize(mask_full.astype(np.uint8), (img_w, img_h),
                                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay = np.zeros_like(vis)
                    overlay[mask_full] = [0, 200, 0]
                    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

                # Hand mask overlay (orange)
                hand_mask_full = load_mask(args.mask_dir, fidx, f"{anchor}_hand_0")
                if hand_mask_full is not None:
                    if hand_mask_full.shape != (img_h, img_w):
                        hand_mask_full = cv2.resize(hand_mask_full.astype(np.uint8), (img_w, img_h),
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay = np.zeros_like(vis)
                    overlay[hand_mask_full] = [0, 140, 255]
                    vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)

                # Use full-res intrinsics for visualization projection
                sx_viz, sy_viz = img_w / pm_w, img_h / pm_h
                fx_viz, fy_viz = fx * sx_viz, fy * sy_viz
                cx_viz, cy_viz = cx * sx_viz, cy * sy_viz

                # Hand projection uses ref-frame intrinsics (matching the computation)
                hfx_viz, hfy_viz = hfx * (img_w / hand_proj_w if ref_k is not None else sx_viz), hfy * (img_h / hand_proj_h if ref_k is not None else sy_viz)
                hcx_viz, hcy_viz = hcx * (img_w / hand_proj_w if ref_k is not None else sx_viz), hcy * (img_h / hand_proj_h if ref_k is not None else sy_viz)

                # Projected hand vertices (cyan = all HaWoR verts)
                all_hand_verts = hand_data[f'{anchor}_vertices'][hi]
                hu_all, hv_all = project_to_pixels(all_hand_verts, hfx_viz, hfy_viz, hcx_viz, hcy_viz, img_w, img_h)
                for pu, pv in zip(hu_all, hv_all):
                    cv2.circle(vis, (pu, pv), 2, (200, 200, 0), -1)
                # Raycast hits on the HaWoR mesh used for h_real (bright yellow)
                hu_hit, hv_hit = project_to_pixels(hand_hits, hfx_viz, hfy_viz, hcx_viz, hcy_viz, img_w, img_h)
                for pu, pv in zip(hu_hit, hv_hit):
                    cv2.circle(vis, (pu, pv), 3, (0, 255, 255), -1)

                # Projected mesh: original (blue) and optimized (red)
                t_cam_orig = fr.get("t_cam_orig", t_cam)
                verts_orig_scale = (mesh_verts * args.scale) @ rot_matrix.T + t_cam_orig
                ou, ov = project_to_pixels(verts_orig_scale, fx_viz, fy_viz, cx_viz, cy_viz, img_w, img_h)
                for pu, pv in zip(ou, ov):
                    cv2.circle(vis, (pu, pv), 1, (255, 0, 0), -1)

                verts_opt = verts_rotated + t_cam * opt_scale
                ou2, ov2 = project_to_pixels(verts_opt, fx_viz, fy_viz, cx_viz, cy_viz, img_w, img_h)
                for pu, pv in zip(ou2, ov2):
                    cv2.circle(vis, (pu, pv), 2, (0, 0, 255), -1)

                # Target point (magenta cross)
                if obj_target[2] > 0:
                    tx = int(fx_viz * obj_target[0] / obj_target[2] + cx_viz)
                    ty = int(fy_viz * obj_target[1] / obj_target[2] + cy_viz)
                    tx = np.clip(tx, 0, img_w - 1)
                    ty = np.clip(ty, 0, img_h - 1)
                    cv2.drawMarker(vis, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

                # Legend
                cv2.putText(vis, f"frame {fidx}  s={opt_scale:.3f}  k={k:.3f}  err={err_before:.3f}->{err_after:.3f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(vis, "blue=orig  red=opt  cyan=hand_verts  yellow=raycast_hits  green=obj_mask  X=target",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cv2.imwrite(os.path.join(viz_dir, f"frame_{fidx:06d}.png"), vis)

        # Update output layout
        obj = output_data["objects"][fr["obj_index"]]
        optimized_t = (t_cam * opt_scale).tolist()
        obj["local_to_scene"]["translation_camera_frame"] = optimized_t
        obj["local_to_scene"]["translation_scale_optimized"] = float(opt_scale)

        per_frame_scales.append({
            "frame_idx": fidx,
            "translation_scale": float(opt_scale),
            "pointmap_scale": float(k),
            "error_before": float(err_before),
            "error_after": float(err_after),
            "obj_target": obj_target.tolist(),
            "mask_pixels": int(n_pixels),
        })

    # Store summary in output
    output_data["translation_scale_optimization"] = {
        "method": "hand_anchored_pointmap",
        "mesh_scale_original": args.scale,
        "mesh_scale": mesh_scale,
        "ref_frame": args.ref_frame,
        "ref_frame_k": ref_k,
        "mask_name": args.mask_name,
        "anchor_hand": args.anchor_hand,
        "per_frame": per_frame_scales,
        # Frames above whose translation_camera_frame was NOT rescaled. They
        # keep the raw tracker value and must not be used as object poses.
        "skipped_frames": sorted(
            {fr["frame_idx"] for fr in frames}
            - {p["frame_idx"] for p in per_frame_scales}
        ),
    }

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    if per_frame_scales:
        scales = [s["translation_scale"] for s in per_frame_scales]
        ks = [s["pointmap_scale"] for s in per_frame_scales]
        errs_before = [s["error_before"] for s in per_frame_scales]
        errs_after = [s["error_after"] for s in per_frame_scales]
        print(f"\nDone. Optimized {len(per_frame_scales)} frames ({skipped} skipped).")
        print(f"  Scale range:    [{min(scales):.4f}, {max(scales):.4f}]")
        print(f"  Scale mean:     {np.mean(scales):.4f}  std: {np.std(scales):.4f}")
        print(f"  PM scale mean:  {np.mean(ks):.4f}  std: {np.std(ks):.4f}")
        print(f"  Error mean:     {np.mean(errs_before):.4f} -> {np.mean(errs_after):.4f}")
    else:
        print(f"\nNo frames optimized ({skipped} skipped).")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
