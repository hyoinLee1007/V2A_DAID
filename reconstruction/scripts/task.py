"""One entry point for the per-task reconstruction commands.

Every command in run.txt repeats the same four paths with the task and object
names spliced in by hand, and each of them is derivable. ``config.json`` in the
task directory already carries what varies:

    {"frame_number": 1, "object_names": ["metalcup"], "anchor_hand": "right"}

and ``frame_number`` is also the only frame whose masks directory holds the
object mesh, so the mesh path follows from it. Globbing for ``*.obj`` instead
would be ambiguous — cupmove's reference frame has ``left_hand_0.obj`` next to
``cup.obj``.

The one thing that is not interchangeable is the scale. ``visualize_3d.py``
wants ``translation_scale_optimization.mesh_scale``, which only exists once
``optimize`` has run (0.142 on cupmove), while ``optimize_translation_scale.py``
reads ``local_to_scene.scale[0]`` (0.394) — different numbers for different
stages. This picks the right one per command rather than leaving a single
MESH_SCALE variable to be pasted into both.

    python task.py metalcupmove paths          # show what was resolved
    python task.py metalcupmove optimize       # optimize_translation_scale.py
    python task.py metalcupmove smooth         # smooth_trajectory.py
    python task.py metalcupmove viz            # visualize_3d.py (newest layout)
    python task.py metalcupmove viz --raw      # ... on the un-optimized layout

Anything after ``--`` is passed through to the wrapped script:

    python task.py metalcupmove viz -- --port 8090 --hands right
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset"

# Newest first: a smoothed layout supersedes the optimized one, which supersedes
# the raw tracker output.
LAYOUTS = [
    "layout_camera_frame_optimized_smooth.json",
    "layout_camera_frame_optimized.json",
    "layout_camera_frame_optimized_raw.json",
    "layout_camera_frame.json",
]

# run_pipeline.sh can leave layout_camera_frame_optimized.json as a symlink to a
# _smooth.json that its smoothing step never wrote. Path.exists() follows the
# link and correctly says no, but the directory listing still shows the name, so
# the failure reads as "the file is right there". _raw is the same optimisation
# before smoothing and is what the smooth step consumes, so it stands in.
_STAGES = {
    "raw": ["layout_camera_frame.json"],
    "optimized": ["layout_camera_frame_optimized.json",
                  "layout_camera_frame_optimized_raw.json"],
    "smooth": ["layout_camera_frame_optimized_smooth.json"],
}


class Task:
    def __init__(self, name: str, dataset: Path = DATASET):
        self.name = name
        self.dir = (dataset / name).resolve()
        if not self.dir.is_dir():
            raise SystemExit(f"no such task directory: {self.dir}")
        cfg_path = self.dir / "config.json"
        if not cfg_path.exists():
            raise SystemExit(f"{cfg_path} not found — run the reconstruction pipeline first")
        cfg = json.load(open(cfg_path))
        self.object = cfg["object_names"][0]
        self.hand = cfg.get("anchor_hand", "right")
        self.frame = int(cfg["frame_number"])

    @property
    def viz_dir(self) -> Path:
        return self.dir / "obj_tracking_out" / self.object / "combined_visualization"

    def layout(self, prefer: str = "newest") -> Path:
        """Newest available layout, or a named stage."""
        names = _STAGES.get(prefer, LAYOUTS)
        for n in names:
            p = self.viz_dir / n
            if p.exists():
                return p
        raise SystemExit(
            f"no layout for stage {prefer!r} under {self.viz_dir}\n"
            f"  present: {sorted(p.name for p in self.viz_dir.glob('layout*.json'))}")

    @property
    def mesh(self) -> Path:
        p = (self.dir / "video_segmentation" / "masks"
             / f"frame_{self.frame:06d}_masks" / self.object / f"{self.object}.obj")
        if not p.exists():
            raise SystemExit(f"object mesh not found: {p}")
        return p

    @property
    def frames_dir(self) -> Path:
        return self.dir / "all_frames"

    @property
    def hand_meshes(self) -> Path:
        return self.dir / self.name / "all_hand_meshes.npz"

    @property
    def intrinsics(self) -> tuple[float, float, float, float] | None:
        """(fx, fy, cx, cy) from the reference frame's intrinsics.txt.

        visualize_3d.py defaults to fx=1346.44, cx=640, 1280x720. metalcupmove
        is 1920x1080 at fx=1544.14, so the defaults put the background image
        and the camera frustum somewhere the geometry is not. run_pipeline.sh
        wrote the real numbers next to the reference frame; use them.
        """
        p = self.dir / f"{self.frame:04d}_intrinsics.txt"
        if not p.exists():
            return None
        try:
            v = [float(x) for x in p.read_text().split()]
        except ValueError:
            return None
        return tuple(v[:4]) if len(v) >= 4 else None

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """(width, height) of the extracted frames, which is what gets rendered."""
        try:
            from PIL import Image
        except ImportError:
            return None
        first = next(iter(sorted(self.frames_dir.glob("*.png"))), None)
        if first is None:
            return None
        with Image.open(first) as im:
            return im.size

    def mesh_scale(self, layout: Path) -> float | None:
        """``visualize_3d.py --scale``; None until `optimize` has been run."""
        d = json.load(open(layout))
        ts = d.get("translation_scale_optimization")
        return float(ts["mesh_scale"]) if ts and "mesh_scale" in ts else None


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("command", choices=("paths", "viz", "optimize", "smooth"))
    p.add_argument("--raw", action="store_true",
                   help="viz: use the un-optimized layout instead of the newest.")
    p.add_argument("--stage", default=None, choices=("raw", "optimized", "smooth"),
                   help="Pick a layout stage explicitly.")
    p.add_argument("--ref-frame", type=int, default=None,
                   help="optimize: reference frame for the pointmap scale. "
                        "Defaults to config.json's frame_number, but that frame "
                        "can have a NaN hand — see repair_hand_nan.py.")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="Arguments after -- go to the wrapped script.")
    args = p.parse_args()
    extra = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    t = Task(args.task)
    stage = args.stage or ("raw" if args.raw else "newest")

    if args.command == "paths":
        print(f"task        {t.name}")
        print(f"object      {t.object}")
        print(f"anchor hand {t.hand}")
        print(f"ref frame   {t.frame}")
        print(f"dir         {t.dir}")
        print(f"frames      {t.frames_dir}")
        print(f"mesh        {t.mesh}")
        print(f"hand meshes {t.hand_meshes}")
        for name in LAYOUTS:
            q = t.viz_dir / name
            mark = "[x]" if q.exists() else ("[!]" if q.is_symlink() else "[ ]")
            note = "  (깨진 심볼릭 링크)" if mark == "[!]" else ""
            print(f"layout      {mark} {name}{note}")
        lay = t.layout(stage)
        print(f"newest      {lay.name}   mesh_scale={t.mesh_scale(lay)}")
        return

    if args.command == "optimize":
        # The raw layout is the input here: optimize writes the optimized one.
        sys.exit(run([sys.executable, HERE / "optimize_translation_scale.py",
                      "--video-dir", t.dir,
                      "--layout-json", t.layout("raw"),
                      "--mesh", t.mesh,
                      "--anchor-hand", t.hand,
                      "--ref-frame", args.ref_frame if args.ref_frame is not None else t.frame,
                      *extra]))

    if args.command == "smooth":
        src = t.layout("optimized")
        dst = src.with_name("layout_camera_frame_optimized_smooth.json")
        sys.exit(run([sys.executable, HERE / "smooth_trajectory.py",
                      "--input", src, "--output", dst, *extra]))

    lay = t.layout(stage)
    scale = t.mesh_scale(lay)
    if scale is None:
        raise SystemExit(
            f"{lay.name} has no translation_scale_optimization.mesh_scale — "
            f"run `python task.py {t.name} optimize` first, or pass --scale yourself")
    cmd = [sys.executable, HERE / "visualize_3d.py",
           "--frames-dir", t.frames_dir,
           "--layout-json", lay,
           "--mesh", t.mesh,
           "--scale", scale,
           "--translation-scale", 1.0,
           "--hand-meshes", t.hand_meshes]

    # Anything the caller passed after `--` wins, so these only fill in.
    K, size = t.intrinsics, t.frame_size
    if K and not any(f in extra for f in ("--fx", "--fy", "--cx", "--cy")):
        cmd += ["--fx", K[0], "--fy", K[1], "--cx", K[2], "--cy", K[3]]
    elif K is None:
        print(f"[warn] no {t.frame:04d}_intrinsics.txt; visualize_3d.py's "
              f"1280x720 defaults will misalign the background image.")
    if size and not any(f in extra for f in ("--width", "--height")):
        cmd += ["--width", size[0], "--height", size[1]]

    sys.exit(run(cmd + extra))


if __name__ == "__main__":
    main()
