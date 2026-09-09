"""Validate the silhouette stage against the pipeline's SAM2 segmentation masks.

The rendered silhouettes S_h,t / S_o,t are what C2Dex Sec. III-A(b) operates on,
but nothing in the paper guarantees the do-as-i-do reconstructions actually
project onto the right pixels. The per-frame SAM2 masks under
video_segmentation/masks/ are an independent 2D observation of where the hand
and object really are, so IoU against them measures how much the candidate
contact region S_c,t can be trusted.

Emits per-frame overlays (hand orange, object blue, S_c red) and an IoU table.

Usage:
    conda activate sam3d-objects
    python -m c2dex.debug_vis --seq-dir ../dataset/cupmove --frames 0,10,20,40,60
    python -m c2dex.debug_vis --seq-dir ../dataset/cupmove --compare-intrinsics
"""

import argparse
import os

import cv2
import numpy as np

from .data import gt_mask_path, load_sequence
from .project import iou, rasterize_silhouette
from .silhouette import compute_silhouettes

COLOR_HAND = np.array([0, 140, 255], dtype=np.float64)  # BGR orange
COLOR_OBJ = np.array([255, 120, 0], dtype=np.float64)  # BGR blue
COLOR_CONTACT = np.array([0, 0, 255], dtype=np.float64)  # BGR red


def _read_gt(seq_dir: str, frame_idx: int, name: str) -> np.ndarray | None:
    path = gt_mask_path(seq_dir, frame_idx, name)
    if path is None:
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127


def overlay(seq, result: dict, i: int) -> np.ndarray | None:
    """Blend S_h, S_o and S_c over the source frame."""
    t = int(result["frames"][i])
    frame_path = os.path.join(seq.seq_dir, "all_frames", f"{t:06d}.png")
    img = cv2.imread(frame_path)
    if img is None:
        return None
    out = img.astype(np.float64)
    for mask, color, alpha in (
        (result["S_h"][i], COLOR_HAND, 0.5),
        (result["S_o"][i], COLOR_OBJ, 0.5),
        (result["S_c"][i], COLOR_CONTACT, 0.6),
    ):
        out[mask] = (1 - alpha) * out[mask] + alpha * color

    uv = result["uv_hand"][i][result["candidates"][i]]
    for u, v in np.rint(uv).astype(int):
        cv2.circle(out, (u, v), 1, (255, 255, 255), -1)
    return out.astype(np.uint8)


def overlay_any_frame(seq, t: int) -> np.ndarray | None:
    """Render frame `t` unconditionally, independent of `seq.frame_valid[t]`.

    `compute_silhouettes` skips a frame entirely once EITHER the hand or the
    object is invalid (frame_valid = hand.valid & obj.valid), so a frame with
    a usable object pose but no hand track disappears from the overlay dir
    along with genuinely-empty frames -- there is no way to tell "nothing was
    ever tracked here" from "the infiller's guess for this frame is right
    there, only excluded by the valid_source filter" without opening this up.

    Renders whichever of S_h/S_o has data (independently -- not gated by the
    other), and stamps a status line so a human can see at a glance whether a
    frame's hand silhouette is a real detection, an infiller hallucination, or
    outright missing:
        TRACKED   -- seq.hand.detected[t] (this frame was actually observed)
        INFERRED  -- hand.infiller_valid[t] but not detected (hallucinated)
        NO HAND   -- infiller itself declined this frame
    and similarly OBJ / NO OBJ for the object pose.
    """
    frame_path = os.path.join(seq.seq_dir, "all_frames", f"{t:06d}.png")
    img = cv2.imread(frame_path)
    if img is None:
        return None
    out = img.astype(np.float64)

    have_hand = bool(seq.hand.infiller_valid[t])
    if have_hand:
        S_h = rasterize_silhouette(
            seq.hand.vertices[t], seq.hand.faces, seq.K_hand[t], seq.width, seq.height
        )
        out[S_h] = 0.5 * out[S_h] + 0.5 * COLOR_HAND

    have_obj = bool(seq.obj.valid[t])
    if have_obj:
        S_o = rasterize_silhouette(
            seq.obj.posed_vertices(t), seq.obj.faces, seq.K_obj[t], seq.width, seq.height
        )
        out[S_o] = 0.5 * out[S_o] + 0.5 * COLOR_OBJ

    if have_hand and have_obj:
        S_c = S_h & S_o
        out[S_c] = 0.4 * out[S_c] + 0.6 * COLOR_CONTACT

    out = out.astype(np.uint8)
    hand_status = (
        "TRACKED" if seq.hand.detected[t]
        else "INFERRED" if seq.hand.infiller_valid[t]
        else "NO HAND"
    )
    obj_status = "OBJ" if seq.obj.valid[t] else "NO OBJ"
    label = f"frame {t}  {hand_status}  {obj_status}"
    color = (0, 255, 0) if hand_status == "TRACKED" else (0, 165, 255) if hand_status == "INFERRED" else (0, 0, 255)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def iou_table(seq, result: dict) -> dict:
    """IoU of each rendered silhouette against its SAM2 mask."""
    hand_name = f"{seq.hand.side}_hand_0"
    rows = []
    for i, t in enumerate(result["frames"]):
        t = int(t)
        if not result["frame_valid"][i]:
            rows.append((t, None, None))
            continue
        gt_h = _read_gt(seq.seq_dir, t, hand_name)
        gt_o = _read_gt(seq.seq_dir, t, seq.obj.name)
        rows.append((
            t,
            iou(result["S_h"][i], gt_h) if gt_h is not None else None,
            iou(result["S_o"][i], gt_o) if gt_o is not None else None,
        ))

    print()
    print(f"{'frame':>6} {'hand IoU':>10} {'obj IoU':>10} {'|S_c|':>8} {'cand':>6}")
    print("-" * 44)
    for i, (t, ih, io) in enumerate(rows):
        fmt = lambda v: f"{v:>10.3f}" if v is not None else f"{'-':>10}"
        if not result["frame_valid"][i]:
            print(f"{t:>6} {fmt(None)} {fmt(None)} {'-':>8} {'-':>6}")
            continue
        print(f"{t:>6} {fmt(ih)} {fmt(io)} {result['S_c'][i].sum():>8} "
              f"{result['candidates'][i].sum():>6}")

    hand_ious = np.array([r[1] for r in rows if r[1] is not None])
    obj_ious = np.array([r[2] for r in rows if r[2] is not None])
    print()
    for label, arr in (("hand", hand_ious), ("object", obj_ious)):
        if arr.size:
            print(f"{label:>6} IoU: mean {arr.mean():.3f}  median {np.median(arr):.3f}  "
                  f"min {arr.min():.3f}  max {arr.max():.3f}")
    bad = [int(r[0]) for r in rows if r[1] is not None and r[1] < 0.15]
    if bad:
        print(f"\n[warn] hand IoU < 0.15 on frames {bad} -- the hand reconstruction "
              "is unreliable there, so any contact recovered on those frames is too.")
    return {"rows": rows, "hand": hand_ious, "obj": obj_ious}


def compare_intrinsics(args) -> None:
    """Quantify what the --intrinsics choice costs, per-source vs unified."""
    results = {}
    for mode in ("per-source", "unified"):
        seq = load_sequence(args.seq_dir, args.side, args.object, mode, args.valid_source)
        res = compute_silhouettes(seq, _parse_frames(args.frames), verbose=False)
        results[mode] = (seq, res)

    seq_p, res_p = results["per-source"]
    seq_u, res_u = results["unified"]

    print()
    print(f"{'frame':>6} | {'obj IoU':>17} | {'|S_c|':>15} | {'candidates':>15}")
    print(f"{'':>6} | {'per-src':>8} {'unified':>8} | {'per-src':>7} {'unified':>7} "
          f"| {'per-src':>7} {'unified':>7}")
    print("-" * 70)
    d_iou, d_cand = [], []
    for i, t in enumerate(res_p["frames"]):
        t = int(t)
        if not (res_p["frame_valid"][i] and res_u["frame_valid"][i]):
            continue
        gt_o = _read_gt(seq_p.seq_dir, t, seq_p.obj.name)
        ip = iou(res_p["S_o"][i], gt_o) if gt_o is not None else np.nan
        iu = iou(res_u["S_o"][i], gt_o) if gt_o is not None else np.nan
        cp, cu = res_p["candidates"][i].sum(), res_u["candidates"][i].sum()
        d_iou.append(ip - iu)
        d_cand.append(int(cp) - int(cu))
        print(f"{t:>6} | {ip:>8.3f} {iu:>8.3f} | {res_p['S_c'][i].sum():>7} "
              f"{res_u['S_c'][i].sum():>7} | {cp:>7} {cu:>7}")

    d_iou, d_cand = np.array(d_iou), np.array(d_cand)
    print()
    print(f"object IoU delta (per-source - unified): mean {np.nanmean(d_iou):+.4f}  "
          f"median {np.nanmedian(d_iou):+.4f}")
    print(f"per-source wins on {int((d_iou > 0).sum())}/{len(d_iou)} frames")
    print(f"candidate-count delta: mean {d_cand.mean():+.1f}  "
          f"max |delta| {np.abs(d_cand).max()}")


def _parse_frames(spec: str | None):
    if not spec:
        return None
    frames = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            frames.extend(range(int(lo), int(hi) + 1))
        else:
            frames.append(int(part))
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--intrinsics", default="per-source", choices=["per-source", "unified"])
    p.add_argument("--valid-source", default="infiller", choices=["infiller", "tracks", "both"],
                   help="See c2dex.data.load_hand (default: infiller).")
    p.add_argument("--side", default=None)
    p.add_argument("--object", default=None)
    p.add_argument("--frames", default=None, help="e.g. '0,10,20' or '0-40' (default: all).")
    p.add_argument("--out-dir", default=None,
                   help="Overlay output dir (default: {seq-dir}/c2dex/overlays).")
    p.add_argument("--no-overlays", action="store_true", help="IoU table only.")
    p.add_argument("--all-frames", action="store_true",
                   help="Render every frame in the sequence unconditionally (ignores "
                        "--frames and frame_valid), independently rendering S_h/S_o "
                        "and labeling each frame TRACKED/INFERRED/NO HAND so hallucinated "
                        "frames are visible instead of silently dropped.")
    p.add_argument("--compare-intrinsics", action="store_true",
                   help="Run both intrinsics modes and diff them.")
    args = p.parse_args()

    if args.compare_intrinsics:
        compare_intrinsics(args)
        return

    seq = load_sequence(args.seq_dir, args.side, args.object, args.intrinsics,
                        args.valid_source)
    print(f"task {seq.task}  intrinsics={seq.intrinsics_mode}  "
          f"valid_source={seq.hand.valid_source}  "
          f"hand f={seq.K_hand[0, 0, 0]:.4f}  "
          f"obj f={seq.K_obj[:, 0, 0].min():.2f}..{seq.K_obj[:, 0, 0].max():.2f}")
    result = compute_silhouettes(seq, _parse_frames(args.frames), verbose=False)
    iou_table(seq, result)

    if args.all_frames:
        out_dir = args.out_dir or os.path.join(seq.seq_dir, "c2dex", "overlays")
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for t in range(seq.n_frames):
            img = overlay_any_frame(seq, t)
            if img is None:
                continue
            cv2.imwrite(os.path.join(out_dir, f"frame_{t:06d}.png"), img)
            n += 1
        print(f"\nWrote {n}/{seq.n_frames} overlays to {out_dir} "
              "(every frame, TRACKED/INFERRED/NO HAND labeled -- none skipped)")
        return

    if not args.no_overlays:
        out_dir = args.out_dir or os.path.join(seq.seq_dir, "c2dex", "overlays")
        os.makedirs(out_dir, exist_ok=True)
        n, stale = 0, 0
        for i in range(len(result["frames"])):
            path = os.path.join(out_dir, f"frame_{int(result['frames'][i]):06d}.png")
            if not result["frame_valid"][i]:
                # Drop an overlay a previous, more permissive run left behind,
                # so the directory always reflects the current settings.
                if os.path.exists(path):
                    os.remove(path)
                    stale += 1
                continue
            img = overlay(seq, result, i)
            if img is None:
                continue
            cv2.imwrite(path, img)
            n += 1
        print(f"\nWrote {n} overlays to {out_dir}"
              + (f" (removed {stale} stale overlays for now-invalid frames)" if stale else ""))


if __name__ == "__main__":
    main()
