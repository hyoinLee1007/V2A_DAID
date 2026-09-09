"""Candidate contact regions S_c,t and the hand vertices that fall inside them.

This is the image-space half of C2Dex Sec. III-A(b):

    "For each frame, we render the initial hand mesh and the posed object mesh
     under the estimated camera. Let S_h,t and S_o,t denote their image-space
     silhouettes; their overlap S_c,t = S_h,t ^ S_o,t defines a candidate
     contact region. We project the hand vertices onto the image plane and
     retain those whose projections lie inside S_c,t."

What this stage deliberately does NOT do, because the paper places it after
this point: casting a ray from the camera centre through each retained vertex
to obtain the object-side observation x_t,i, and the normal-consistency filter
w^n_t,i = -(n^h_t,i . n^o_t,i) / (||n^h_t,i|| ||n^o_t,i||) > gamma_n that yields
the retained index set C_t. This stage emits the candidate set those steps
consume.

Usage:
    conda activate sam3d-objects
    python -m c2dex.silhouette --seq-dir ../dataset/cupmove
    python -m c2dex.silhouette --seq-dir ../dataset/cupmove --intrinsics unified
"""

import argparse
import os

import numpy as np

from .data import load_sequence
from .project import points_in_mask, project_points, rasterize_silhouette


def compute_silhouettes(seq, frames=None, verbose: bool = True) -> dict:
    """Compute S_h,t, S_o,t, S_c,t and the candidate hand-vertex set.

    Returns a dict of stacked per-frame arrays over `frames` (default: all).
    Frames whose hand or object pose is invalid still get an entry, with empty
    masks and no candidates, so that array index == frame index throughout.
    """
    if frames is None:
        frames = range(seq.n_frames)
    frames = list(frames)

    n_hand_verts = seq.hand.vertices.shape[1]
    shape = (len(frames), seq.height, seq.width)

    S_h = np.zeros(shape, dtype=bool)
    S_o = np.zeros(shape, dtype=bool)
    S_c = np.zeros(shape, dtype=bool)
    uv_hand = np.full((len(frames), n_hand_verts, 2), np.nan)
    hand_in_front = np.zeros((len(frames), n_hand_verts), dtype=bool)
    candidates = np.zeros((len(frames), n_hand_verts), dtype=bool)
    valid = np.zeros(len(frames), dtype=bool)

    for i, t in enumerate(frames):
        if not seq.frame_valid[t]:
            if verbose:
                reason = []
                if not seq.hand.detected[t]:
                    reason.append("hand not detected (infilled)")
                elif not seq.hand.infiller_valid[t]:
                    reason.append("hand marked invalid")
                if not seq.obj.valid[t]:
                    reason.append("object pose missing or behind camera")
                print(f"  frame {t:4d}: skipped ({', '.join(reason)})")
            continue

        valid[i] = True
        hand_cam = seq.hand.vertices[t]
        obj_cam = seq.obj.posed_vertices(t)

        S_h[i] = rasterize_silhouette(
            hand_cam, seq.hand.faces, seq.K_hand[t], seq.width, seq.height
        )
        S_o[i] = rasterize_silhouette(
            obj_cam, seq.obj.faces, seq.K_obj[t], seq.width, seq.height
        )
        S_c[i] = S_h[i] & S_o[i]

        # Hand vertices are projected under the hand's own intrinsics, matching
        # the silhouette they were rasterized into.
        uv, in_front = project_points(hand_cam, seq.K_hand[t])
        uv_hand[i] = uv
        hand_in_front[i] = in_front
        candidates[i] = points_in_mask(uv, S_c[i])

    return {
        "frames": np.asarray(frames, dtype=np.int32),
        "S_h": S_h,
        "S_o": S_o,
        "S_c": S_c,
        "uv_hand": uv_hand,
        "hand_in_front": hand_in_front,
        "candidates": candidates,
        "frame_valid": valid,
    }


def save(result: dict, seq, path: str) -> None:
    """Write the stage output, bit-packing the three mask stacks."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        frames=result["frames"],
        frame_valid=result["frame_valid"],
        candidates=result["candidates"],
        uv_hand=result["uv_hand"].astype(np.float32),
        hand_in_front=result["hand_in_front"],
        S_h_packed=np.packbits(result["S_h"], axis=-1),
        S_o_packed=np.packbits(result["S_o"], axis=-1),
        S_c_packed=np.packbits(result["S_c"], axis=-1),
        mask_shape=np.array([seq.height, seq.width], dtype=np.int32),
        intrinsics_mode=seq.intrinsics_mode,
        hand_side=seq.hand.side,
        object_name=seq.obj.name,
        mesh_scale=seq.obj.mesh_scale,
        K_hand=seq.K_hand,
        K_obj=seq.K_obj,
    )


def load(path: str) -> dict:
    """Inverse of `save`, unpacking the mask stacks."""
    d = dict(np.load(path, allow_pickle=False))
    h, w = d.pop("mask_shape")
    for key in ("S_h", "S_o", "S_c"):
        packed = d.pop(f"{key}_packed")
        d[key] = np.unpackbits(packed, axis=-1, count=w).astype(bool).reshape(-1, h, w)
    return d


def report(result: dict, seq) -> None:
    """Per-frame summary of the candidate contact region."""
    print()
    print(f"{'frame':>6} {'|S_h|':>8} {'|S_o|':>8} {'|S_c|':>8} {'cand':>6}  {'f_obj':>8}")
    print("-" * 52)
    for i, t in enumerate(result["frames"]):
        if not result["frame_valid"][i]:
            print(f"{t:>6} {'-':>8} {'-':>8} {'-':>8} {'-':>6}  {'-':>8}")
            continue
        print(
            f"{t:>6} {result['S_h'][i].sum():>8} {result['S_o'][i].sum():>8} "
            f"{result['S_c'][i].sum():>8} {result['candidates'][i].sum():>6}  "
            f"{seq.K_obj[t][0, 0]:>8.2f}"
        )

    ok = result["frame_valid"]
    if not ok.any():
        print("\nNo valid frames.")
        return
    n_cand = result["candidates"].sum(axis=1)
    area = result["S_c"].reshape(len(ok), -1).sum(axis=1)
    print()
    print(f"valid frames        : {ok.sum()}/{len(ok)}")
    print(f"|S_c| px            : min {area[ok].min()}  median {np.median(area[ok]):.0f}  max {area[ok].max()}")
    print(f"candidate vertices  : min {n_cand[ok].min()}  median {np.median(n_cand[ok]):.0f}  max {n_cand[ok].max()}")
    empty = result["frames"][ok & (n_cand == 0)].tolist()
    if empty:
        print(f"frames with no candidate vertex: {empty}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seq-dir", required=True, help="Reconstruction output dir for one video.")
    p.add_argument("--intrinsics", default="per-source", choices=["per-source", "unified"],
                   help="Camera model. See c2dex.data.load_intrinsics (default: per-source).")
    p.add_argument("--valid-source", default="infiller", choices=["infiller", "tracks", "both"],
                   help="Which frames count as usable. 'infiller' (default) trusts "
                        "all_hand_meshes.npz, which marks infilled frames valid; "
                        "'tracks' uses cam_space detection coverage; 'both' intersects them.")
    p.add_argument("--side", default=None, help="Hand side; defaults to config.json anchor_hand.")
    p.add_argument("--object", default=None, help="Object name; defaults to config.json object_names[0].")
    p.add_argument("--frames", default=None,
                   help="Frame subset, e.g. '0,10,20' or '0-40' (default: all).")
    p.add_argument("--output", default=None,
                   help="Output .npz (default: {seq-dir}/c2dex/contact_candidates.npz).")
    p.add_argument("--no-save", action="store_true", help="Report only, write nothing.")
    args = p.parse_args()

    seq = load_sequence(args.seq_dir, args.side, args.object, args.intrinsics,
                        args.valid_source)
    infilled = np.where(~seq.hand.detected)[0]
    print(f"task            : {seq.task}  ({seq.n_frames} frames, {seq.width}x{seq.height})")
    print(f"hand            : {seq.hand.side}, {seq.hand.vertices.shape[1]} verts, "
          f"{len(seq.hand.faces)} faces, {seq.hand.valid.sum()} valid frames "
          f"(valid_source={seq.hand.valid_source})")
    print(f"  detected      : {seq.hand.detected.sum()}/{seq.n_frames}"
          + (f"  infilled frames: {infilled.tolist()}" if infilled.size else ""))
    print(f"object          : {seq.obj.name}, {len(seq.obj.canonical_vertices)} verts, "
          f"{len(seq.obj.faces)} faces, mesh_scale={seq.obj.mesh_scale:.8f}, "
          f"{seq.obj.valid.sum()} valid frames")
    print(f"intrinsics      : {seq.intrinsics_mode}  "
          f"hand f={seq.K_hand[0, 0, 0]:.4f}  "
          f"obj f={seq.K_obj[:, 0, 0].min():.2f}..{seq.K_obj[:, 0, 0].max():.2f}")

    frames = None
    if args.frames:
        frames = []
        for part in args.frames.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                frames.extend(range(int(lo), int(hi) + 1))
            else:
                frames.append(int(part))

    result = compute_silhouettes(seq, frames)
    report(result, seq)

    if not args.no_save:
        out = args.output or os.path.join(seq.seq_dir, "c2dex", "contact_candidates.npz")
        save(result, seq, out)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
