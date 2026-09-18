#!/usr/bin/env python3
"""Find out why SAM3 wrote empty hand masks for a run of frames.

iphone_video's left_hand_0.png is empty (0 px) on frames 43-84 although the
hand is plainly visible and nearly identical to frame 42, which has a mask.
Frame 43 is also the prompt frame run_pipeline.sh gave SAM3. Two explanations
fit, and they need different fixes:

1. **The detector does not see a "left hand" there.** Scores for the text
   prompt fall under score_threshold_detection (0.5) / new_det_thresh (0.7) on
   43-84 — e.g. the pose reads as a right hand. Fix: a different prompt
   ("hand") or a click prompt.
2. **The tracker dropped the track.** build_sam3_video_model's default
   (apply_temporal_disambiguation=True) runs hotstart: a track unmatched by
   detections on 8 of its first 15 frames is removed, and the removed id is
   hidden from every frame still in the 15-frame buffer. A track born at the
   prompt frame is exactly the one exposed to that. Fix: prompt on a frame
   where detection is strong (the gap should move with the prompt frame).

This script separates them without touching the dataset's masks:

    python diagnose_sam3_gap.py --task iphone_video
    python diagnose_sam3_gap.py --task iphone_video --prompt-frames 43 42 85 --skip-detector

Needs the GPU (~ the same memory run_sam3_video.py uses). Run it in the env
run_pipeline.sh uses for SAM3.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="iphone_video")
    p.add_argument("--dataset-dir", default=str(HERE.parent / "dataset"))
    p.add_argument("--prompt", default=None,
                   help="text prompt; default '<anchor_hand> hand' from config.json")
    p.add_argument("--prompt-frames", type=int, nargs="+", default=None,
                   help="prompt frames to trace; default config.json's frame_number")
    p.add_argument("--detector-frames", type=int, nargs="+",
                   default=[38, 40, 42, 43, 44, 46, 50, 60, 70, 80, 83, 84, 85, 86, 90])
    p.add_argument("--detector-prompts", nargs="+",
                   default=["left hand", "right hand", "hand"])
    p.add_argument("--skip-detector", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--out", default=None, help="JSON report; default <task>/sam3_gap_report.json")
    return p.parse_args()


def runs(frames: list[int]) -> str:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in out) or "none"


def detector_scores(frames_dir: Path, frame_ids: list[int], prompts: list[str]) -> dict:
    """Per-frame best score of the image detector, one prompt at a time."""
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    # Threshold 0 so a weak detection still reports its score instead of vanishing.
    proc = Sam3Processor(model, confidence_threshold=0.0)
    table: dict[int, dict[str, float]] = {}
    for f in frame_ids:
        img = Image.open(frames_dir / f"{f:06d}.png").convert("RGB")
        state = proc.set_image(img)
        table[f] = {}
        for prompt in prompts:
            proc.reset_all_prompts(state)
            out = proc.set_text_prompt(prompt, state)
            s = out.get("scores")
            table[f][prompt] = float(s.max()) if s is not None and len(s) else 0.0
    del proc, model
    torch.cuda.empty_cache()
    return table


def video_trace(video: Path, prompt: str, prompt_frame: int) -> dict:
    """Replay run_sam3_video.py's text-prompt run, recording instead of saving."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()
    sid = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(video)))["session_id"]
    resp = predictor.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=prompt_frame,
        text=prompt, obj_id=1))
    po = resp["outputs"]
    prompt_out = {"ids": [int(i) for i in po["out_obj_ids"]],
                  "probs": [round(float(p), 4) for p in po["out_probs"]]}

    passes: list[list[dict]] = [[]]
    for r in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=sid)):
        fi, out = int(r["frame_index"]), r["outputs"]
        cur = passes[-1]
        # "both" = forward from the prompt frame, then backward from it again.
        if len(passes) == 1 and cur and fi < cur[-1]["frame"]:
            passes.append([])
            cur = passes[-1]
        cur.append({
            "frame": fi,
            "ids": [int(i) for i in out["out_obj_ids"]],
            "probs": [round(float(p), 4) for p in out["out_probs"]],
            "area": [int(m.sum()) for m in out["out_binary_masks"]],
        })

    predictor.handle_request(request=dict(type="close_session", session_id=sid))
    predictor.shutdown()
    torch.cuda.empty_cache()

    # run_sam3_video.py keeps the last output per frame, so the backward pass
    # overwrites the prompt frame. Reproduce that to compare with the masks on disk.
    merged: dict[int, dict] = {}
    for p in passes:
        for row in p:
            merged[row["frame"]] = row
    return {
        "prompt_frame": prompt_frame,
        "prompt_frame_output": prompt_out,
        "passes": {("forward" if i == 0 else "backward"): p for i, p in enumerate(passes)},
        "merged_empty_frames": sorted(f for f, r in merged.items() if not r["ids"]),
    }


def main() -> None:
    a = parse_args()
    task_dir = Path(a.dataset_dir) / a.task
    cfg = json.load(open(task_dir / "config.json"))
    hand = cfg.get("anchor_hand", "right")
    prompt = a.prompt or f"{hand} hand"
    prompt_frames = a.prompt_frames or [int(cfg["frame_number"])]
    report: dict = {"task": a.task, "prompt": prompt}

    # What is on disk now, for comparison.
    import cv2
    on_disk = []
    for d in sorted((task_dir / "video_segmentation" / "masks").glob("frame_*_masks")):
        m = cv2.imread(str(d / f"{hand}_hand_0.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None and not (m > 127).any():
            on_disk.append(int(d.name.split("_")[1]))
    report["empty_on_disk"] = on_disk
    print(f"empty {hand}_hand_0 masks on disk: {runs(on_disk)}")

    if not a.skip_detector:
        print("\n== image detector, best score per prompt ==")
        table = detector_scores(task_dir / "all_frames", a.detector_frames, a.detector_prompts)
        report["detector"] = table
        print("frame  " + "  ".join(f"{p:>11s}" for p in a.detector_prompts))
        for f, row in table.items():
            flag = "  <- empty on disk" if f in on_disk else ""
            print(f"{f:5d}  " + "  ".join(f"{row[p]:11.3f}" for p in a.detector_prompts) + flag)
        print("(video thresholds: score_threshold_detection 0.5, new_det_thresh 0.7)")

    if not a.skip_video:
        report["video"] = []
        video = task_dir / f"{a.task}.mp4"
        for pf in prompt_frames:
            print(f"\n== video trace: '{prompt}' prompted on frame {pf} ==")
            tr = video_trace(video, prompt, pf)
            report["video"].append(tr)
            print(f"prompt frame output: {tr['prompt_frame_output']}")
            for name, rows in tr["passes"].items():
                empty = [r["frame"] for r in rows if not r["ids"]]
                ids = sorted({i for r in rows for i in r["ids"]})
                print(f"  {name:8s} frames {len(rows):3d}  empty: {runs(empty)}  track ids seen: {ids}")
            print(f"  merged empty (what would be written): {runs(tr['merged_empty_frames'])}")

    out = Path(a.out) if a.out else task_dir / "sam3_gap_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
