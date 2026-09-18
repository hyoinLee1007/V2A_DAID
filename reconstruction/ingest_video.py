#!/usr/bin/env python3
"""Download or import one clip into dataset/ under a name the pipeline accepts.

The pipeline's only naming constraint is that ``--task`` equals the video's
basename up to the FIRST DOT: run_pipeline.sh derives VIDEO_NAME with
``${VIDEO_BASENAME%%.*}`` and HaWoR derives its seq folder with
``basename(video).split('.')[0]``, and process_dataset then reads
``{raw_dir}/{task}/all_hand_meshes.npz``. The containing directory's name is
never read by anything, so the dataset/<name>/<name>.mp4 convention is a
convenience, not a requirement — this keeps it anyway so the name has to be
typed once.

Web filenames break that constraint constantly ("How.to.Pour.Coffee.1080p.mp4"
becomes the task "How"), and the failure is silent: process_dataset falls back
to a glob over raw_dir and can pick up a different clip's track. So the name is
slugified here, once, at the only point where it is still cheap to change.

    python ingest_video.py 'https://youtu.be/XXXX' --start 1:23 --duration 6
    python ingest_video.py /data/clip.mp4 --name pourcoffee

Two things this records that the pipeline currently guesses:

- **Frame rate.** ``ref_dt`` in config/override/do_as_i_do.yaml is hardcoded to
  0.0333 (30 FPS) for every task, but the clips already in dataset/ run at 16,
  24, 29.97, 30 and 50 FPS. At 50 the reference is replayed 1.67x too slow, at
  16 it is 1.9x too fast. config.py:690 already overrides ref_dt from
  task_info.json — nothing writes it. meta.json here carries the measured fps
  so that hole can be closed; --fps resamples instead, if you would rather
  every clip really were 30.

- **Disk.** Step 0 writes every frame as a PNG, and get_pointmap_dir.py writes
  an .npy beside each one. Measured: appleinput is 5.9 s at 24 FPS and its
  all_frames is 1.6 GB. A clip is charged roughly 270 MB per second of 720p
  footage, so length is the binding constraint at web scale, not download time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# yt-dlp lives in the sam3 env; the pipeline's own step 0 uses that env's ffmpeg
# for the same reason.
YTDLP_FALLBACK = Path.home() / "miniconda3/envs/sam3/bin/yt-dlp"
# Above this, warn about all_frames rather than refuse — a long clip is
# sometimes what you want, but it should not be a surprise.
WARN_SECONDS = 12.0
MB_PER_SECOND_720P = 270.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video URL (yt-dlp) or a local video file")
    p.add_argument("--name", default=None,
                   help="task name; default is the slugified title or filename")
    p.add_argument("--dataset-dir", default=str(HERE / "dataset"))
    p.add_argument("--start", default=None, help="trim start, e.g. 1:23 or 83.5")
    p.add_argument("--end", default=None, help="trim end (mutually exclusive with --duration)")
    p.add_argument("--duration", default=None, help="trim length in seconds")
    p.add_argument("--fps", type=float, default=None,
                   help="resample to this frame rate; default keeps the source rate "
                        "and records it in meta.json")
    p.add_argument("--object", default=None, help="object name, recorded for the run command")
    p.add_argument("--hand", default=None, choices=("left", "right", "bimanual"))
    p.add_argument("--frame", type=int, default=None,
                   help="reference frame index, recorded for the run command")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing dataset/<name> instead of suffixing")
    p.add_argument("--keep-download", action="store_true",
                   help="keep the raw download beside the clip")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def slugify(text: str) -> str:
    """A name safe for the task/dot constraint above: [a-z0-9_], no dots."""
    ascii_text = (unicodedata.normalize("NFKD", text)
                  .encode("ascii", "ignore").decode("ascii"))
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    slug = re.sub(r"_{2,}", "_", slug)
    if len(slug) >= 3:
        return slug
    # Non-Latin titles are common at web scale and collapse to nothing here, so
    # every one of them would land on the same name and be told it collided. A
    # short hash of the original keeps distinct clips distinct; meta.json keeps
    # the real title.
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{h}" if slug else f"clip_{h}"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def ytdlp() -> str:
    exe = shutil.which("yt-dlp") or (str(YTDLP_FALLBACK) if YTDLP_FALLBACK.exists() else None)
    if exe is None:
        raise SystemExit("yt-dlp not found on PATH or in the sam3 env")
    return exe


def probe(path: Path) -> dict:
    """fps as a float, plus duration and size, from the container's own metadata."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    st = (d.get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    nb = st.get("nb_frames")
    return {
        "fps": round(fps, 6),
        "width": st.get("width"),
        "height": st.get("height"),
        "duration": float(d.get("format", {}).get("duration", 0.0) or 0.0),
        "n_frames": int(nb) if nb and str(nb).isdigit() else None,
    }


def fetch(source: str, work: Path) -> tuple[Path, str | None, str | None]:
    """(local file, title, url). A local source is used in place, not copied."""
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve(), local.stem, None

    work.mkdir(parents=True, exist_ok=True)
    exe = ytdlp()
    title = subprocess.run([exe, "--get-title", "--no-warnings", source],
                           check=True, capture_output=True, text=True).stdout.strip()
    # A fixed output name: yt-dlp's own template would reintroduce the dots and
    # spaces this script exists to remove, and the file is renamed anyway.
    tmpl = str(work / "download.%(ext)s")
    run([exe, "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b", "-o", tmpl, source])
    hits = sorted(work.glob("download.*"))
    if not hits:
        raise SystemExit(f"yt-dlp produced no file in {work}")
    return hits[0], title or None, source


def unique_dir(root: Path, slug: str, force: bool) -> tuple[Path, str]:
    """dataset/<slug>, suffixed _2, _3 ... unless it is free or --force."""
    d = root / slug
    if force or not d.exists():
        return d, slug
    for i in range(2, 1000):
        cand = root / f"{slug}_{i}"
        if not cand.exists():
            print(f"[note] {d.name} exists; using {cand.name}")
            return cand, cand.name
    raise SystemExit(f"Too many collisions on {slug}")


def build_ffmpeg(src: Path, dst: Path, a: argparse.Namespace) -> list[str]:
    """Re-encode: the pipeline seeks by frame index, so the clip must start at
    frame 0 with no leading edit list, and a stream copy of a trimmed range
    starts at the previous keyframe instead."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if a.start:
        cmd += ["-ss", a.start]
    cmd += ["-i", str(src)]
    if a.end:
        cmd += ["-to", a.end]
    elif a.duration:
        cmd += ["-t", a.duration]
    if a.fps:
        cmd += ["-r", str(a.fps)]
    # -an: nothing downstream reads audio, and it would survive into every copy.
    cmd += ["-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    return cmd


def main() -> None:
    a = parse_args()
    if a.end and a.duration:
        raise SystemExit("--end and --duration are mutually exclusive")

    root = Path(a.dataset_dir).resolve()
    work = root / ".ingest_tmp"
    src, title, url = fetch(a.source, work)
    src_info = probe(src)
    print(f"\nsource: {src}")
    print(f"  {src_info['width']}x{src_info['height']}  {src_info['fps']:g} fps  "
          f"{src_info['duration']:.2f} s")

    slug = slugify(a.name or title or src.stem)
    out_dir, slug = unique_dir(root, slug, a.force)
    dst = out_dir / f"{slug}.mp4"
    print(f"\nname:   {slug}\ntarget: {dst}")

    cmd = build_ffmpeg(src, dst, a)
    if a.dry_run:
        print("\n[dry-run] would run:\n  $", " ".join(cmd))
        if url and not a.keep_download:
            shutil.rmtree(work, ignore_errors=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    run(cmd)
    info = probe(dst)

    est_mb = info["duration"] * MB_PER_SECOND_720P
    print(f"\nclip:   {info['width']}x{info['height']}  {info['fps']:g} fps  "
          f"{info['duration']:.2f} s  ({dst.stat().st_size / 1e6:.1f} MB)")
    if info["duration"] > WARN_SECONDS:
        print(f"[warn] {info['duration']:.1f} s will expand to roughly "
              f"{est_mb / 1000:.1f} GB of all_frames PNGs + pointmap .npy. "
              f"Trim with --start/--duration if that is not intended.")

    meta = {
        "task": slug,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": url,
        "source_title": title,
        "source_path": None if url else str(src),
        "source": src_info,
        "clip": info,
        "trim": {"start": a.start, "end": a.end, "duration": a.duration},
        "resampled_fps": a.fps,
        # config.py:690 overrides ref_dt from task_info.json but nothing writes
        # it, so every task silently uses the 30 FPS default. This is the value
        # it should carry for this clip.
        "ref_dt": round(1.0 / info["fps"], 6) if info["fps"] else None,
        "object_names": [a.object] if a.object else None,
        "anchor_hand": a.hand,
        "frame_number": a.frame,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"        {out_dir / 'meta.json'}")

    if url and not a.keep_download:
        shutil.rmtree(work, ignore_errors=True)

    obj = a.object or "OBJECT"
    hand = a.hand or "right"
    frame = a.frame if a.frame is not None else 0
    ret_dir = HERE.parent / "retargeting"
    rel = os.path.relpath(out_dir, ret_dir)
    # A dataset kept outside the repo relativises into a wall of "..", which is
    # worse than the absolute path it came from.
    raw_dir = rel if not rel.startswith("../../") else str(out_dir)
    print(f"\nnext:\n  cd {HERE}\n"
          f"  ./run_pipeline.sh {dst} {frame} {obj} {hand}\n"
          f"  cd {ret_dir}\n"
          f"  python launch.py --task {slug} --raw-dir {raw_dir}")
    if meta["ref_dt"] and abs(meta["ref_dt"] - 0.0333) > 0.002:
        skew = meta["ref_dt"] / 0.0333
        how = f"{skew:.2f}x too fast" if skew > 1 else f"{1 / skew:.2f}x too slow"
        print(f"\n[warn] this clip is {info['fps']:g} FPS, so ref_dt should be "
              f"{meta['ref_dt']:.4f}. do_as_i_do.yaml says 0.0333, which replays "
              f"the reference {how}. meta.json carries the right value.")


if __name__ == "__main__":
    main()
