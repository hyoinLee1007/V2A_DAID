#!/usr/bin/env python3
"""Serve a contact_heatmap.npz in the browser, colored by confidence.

The two viewers already here both need something extra: visualize_contact_heatmap.py
opens an open3d window (so a display, and the sam3d-objects env — retargeting has no
open3d), and visualize_contact_heatmap_glb.py writes a .glb you then open somewhere
else. This one serves the mesh over a port, so it works headless and over SSH, and
runs in either env (viser + trimesh + matplotlib are in both).

    python viz_heatmap_viser.py --npz heatmap_metalcupmove_side_out --port 8082

--npz takes the .npz itself or a directory holding contact_heatmap.npz. It prints
the summary and exits with --summary-only.

Scalars it can color by:
  contact_confidence  H in [0,1], the term's H(x)
  geodesic_distance   metres from the nearest seed; inf where the mesh is
                      disconnected from every seed, which is worth seeing — a
                      second shell reads as "no contact" for the same reason
  binary_contact      the stored H > 0.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
import viser

SCALARS = ("contact_confidence", "geodesic_distance", "binary_contact")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True,
                   help="contact_heatmap.npz, or a directory containing one")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--colormap", default="viridis")
    p.add_argument("--summary-only", action="store_true",
                   help="print the summary and exit without serving")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        hits = sorted(p.glob("*.npz"))
        if not hits:
            raise SystemExit(f"No .npz under {p}")
        named = [h for h in hits if h.name == "contact_heatmap.npz"]
        p = named[0] if named else hits[0]
        if len(hits) > 1:
            print(f"[note] {len(hits)} npz files in the directory; using {p.name}")
    if not p.exists():
        raise SystemExit(f"Not found: {p}")
    return p


def summarize(d, path: Path) -> None:
    """Every key with its shape and range, then the numbers worth reading."""
    print(f"\n=== {path} ===")
    for k in d.files:
        a = d[k]
        line = f"  {k:22s} {str(a.shape):15s} {a.dtype}"
        if a.ndim == 0:
            line += f"  = {a}"
        elif a.size and np.issubdtype(a.dtype, np.number):
            fin = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
            if fin.size:
                line += f"  [{fin.min():.4g}, {fin.max():.4g}]"
            n_bad = int(a.size - fin.size)
            if n_bad:
                line += f"  ({n_bad} non-finite)"
        print(line)

    if "contact_confidence" not in d.files:
        return
    H = d["contact_confidence"]
    n = H.size
    print(f"\n  vertices              {n}")
    for t in (0.0, 0.25, 0.5, 0.68, 0.9):
        m = int((H > t).sum())
        print(f"  H > {t:<4}             {m:6d}  ({100 * m / n:5.2f}%)")
    if "seed_vertex_indices" in d.files:
        print(f"  seeds                 {d['seed_vertex_indices'].size}")
    # contact_region_max_points keeps the top-N by confidence, so the N-th
    # largest H is the cutoff those configs actually apply. 300 is what
    # do_as_i_do.yaml uses.
    for cap in (300, 2000):
        if cap < n:
            print(f"  H at rank {cap:<5}       {np.partition(H, n - cap)[n - cap]:.4f}"
                  f"   (cutoff if contact_region_max_points={cap})")


def colorize(scalar: np.ndarray, cmap_name: str, lo: float, hi: float) -> np.ndarray:
    """(N,3) uint8. Non-finite entries go flat grey rather than to an end of the map."""
    cmap = matplotlib.colormaps[cmap_name]
    finite = np.isfinite(scalar)
    t = np.zeros_like(scalar, dtype=np.float64)
    if hi > lo:
        t[finite] = np.clip((scalar[finite] - lo) / (hi - lo), 0.0, 1.0)
    rgb = (cmap(t)[:, :3] * 255).astype(np.uint8)
    rgb[~finite] = (110, 110, 110)
    return rgb


def submesh(verts: np.ndarray, faces: np.ndarray, keep_vert: np.ndarray):
    """Faces whose three vertices all pass, reindexed onto the kept vertices."""
    keep_face = keep_vert[faces].all(axis=1)
    if not keep_face.any():
        return None, None, None
    f = faces[keep_face]
    used = np.unique(f)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[f], used


def main() -> None:
    a = parse_args()
    path = resolve(a.npz)
    d = np.load(path, allow_pickle=True)
    summarize(d, path)
    if a.summary_only:
        return

    verts = np.asarray(d["vertices"], np.float64)
    faces = np.asarray(d["faces"], np.int32)
    # Center on the mesh so orbiting turns around the object, not the origin.
    verts = verts - verts.mean(axis=0)
    fields = {k: np.asarray(d[k], np.float64) for k in SCALARS if k in d.files}
    if not fields:
        raise SystemExit(f"{path.name} has none of {SCALARS}; nothing to color by.")
    seeds = (verts[np.asarray(d["seed_vertex_indices"], np.int64)]
             if "seed_vertex_indices" in d.files else None)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))

    server = viser.ViserServer(host=a.host, port=a.port)

    field = server.gui.add_dropdown("Color by", tuple(fields),
                                 initial_value=next(iter(fields)))
    cmap = server.gui.add_dropdown(
        "Colormap", ("viridis", "turbo", "inferno", "magma", "coolwarm"),
        initial_value=a.colormap)
    thresh = server.gui.add_slider("Threshold", 0.0, 1.0, 0.01, 0.0)
    hide = server.gui.add_checkbox("Hide below threshold", False)
    show_seeds = server.gui.add_checkbox("Show seeds", seeds is not None,
                                         disabled=seeds is None)
    # The npz carries no up axis — it is whatever frame the SAM-3D
    # reconstruction produced — so make it switchable instead of guessing.
    up = server.gui.add_dropdown("Up axis", ("+z", "-z", "+y", "-y", "+x", "-x"),
                                 initial_value="+y")
    stat = server.gui.add_text("Above threshold", "", disabled=True)

    def render(_=None) -> None:
        s = fields[field.value]
        finite = np.isfinite(s)
        # Each field gets its own range: confidence is [0,1], geodesic distance
        # is metres and its inf entries must not set the top of the scale.
        lo, hi = (0.0, 1.0) if field.value != "geodesic_distance" else (
            0.0, float(s[finite].max()) if finite.any() else 1.0)
        # The slider is a fraction of the range, so it reads the same either way.
        cut = lo + thresh.value * (hi - lo)

        v, f, used = verts, faces, np.arange(len(verts))
        keep = finite & (s >= cut)
        if hide.value:
            v, f, used = submesh(verts, faces, keep)
            if v is None:
                stat.value = "0 vertices — threshold too high"
                server.scene.reset()
                return

        colors = colorize(s[used], cmap.value, lo, hi)
        if not hide.value:
            # Below the cut stays visible but goes grey, so the threshold reads
            # against the object's actual shape.
            colors[~keep] = (95, 95, 95)

        mesh = trimesh.Trimesh(v, f, vertex_colors=colors, process=False)
        server.scene.add_mesh_trimesh("/heatmap", mesh)

        if seeds is not None and show_seeds.value:
            server.scene.add_point_cloud(
                "/seeds", seeds, np.tile(np.array([[255, 40, 40]], np.uint8),
                                         (len(seeds), 1)),
                point_size=diag * 0.012)
        else:
            server.scene.remove_by_name("/seeds")

        n_keep = int(keep.sum())
        stat.value = f"{n_keep} / {len(verts)}  ({100 * n_keep / len(verts):.2f}%)"

    for h in (field, cmap, thresh, hide, show_seeds):
        h.on_update(render)
    up.on_update(lambda _: server.scene.set_up_direction(up.value))
    server.scene.set_up_direction(up.value)
    render()

    print(f"\nviser on http://localhost:{a.port}   (ctrl-c to stop)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
