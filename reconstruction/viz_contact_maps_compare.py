"""Export several contact heatmaps side by side as one GLB for eyeball comparison.

Each npz is drawn as its own copy of the object mesh, coloured by
``contact_confidence`` (grey -> red). All copies share the same colour scale, so
a redder patch really means higher confidence and not just a different
normalisation.

Because every map produced by this project is indexed in ``visual.obj`` vertex
order, the copies are directly comparable vertex-for-vertex — that is the whole
point of keeping the ordering fixed.

    python viz_contact_maps_compare.py \
        heatmap_out/contact_heatmap_objorder.npz:manual \
        heatmap_out/contact_heatmap_choir.npz:choir-all \
        heatmap_out/contact_heatmap_choir_grasp.npz:choir-grasp \
        heatmap_out/contact_heatmap_choir_smooth.npz:choir-smooth
"""

from __future__ import annotations

import argparse

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("maps", nargs="+",
                   help="Each entry is 'path.npz' or 'path.npz:label'.")
    p.add_argument("--out", default="heatmap_out/contact_maps_compare.glb")
    p.add_argument("--gap", type=float, default=1.8,
                   help="Spacing between copies, in object bbox widths.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene = trimesh.Scene()
    lo = np.array([170.0, 170.0, 170.0])
    hi = np.array([230.0, 40.0, 40.0])

    entries = []
    for spec in args.maps:
        path, _, label = spec.partition(":")
        d = np.load(path)
        entries.append((label or path.split("/")[-1], d))

    span = None
    for i, (label, d) in enumerate(entries):
        V = np.asarray(d["vertices"], float)
        F = np.asarray(d["faces"], np.int64)
        H = np.clip(np.asarray(d["contact_confidence"], float), 0.0, 1.0)
        if span is None:
            span = float(np.ptp(V, 0).max()) * args.gap
        rgb = lo[None, :] * (1 - H[:, None]) + hi[None, :] * H[:, None]
        rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255.0)], 1).astype(np.uint8)
        m = trimesh.Trimesh(vertices=V + [i * span, 0, 0], faces=F, process=False)
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
        scene.add_geometry(m)
        print("%-16s  H>0.5: %5d verts (%.1f%%)   H>0.9: %5d   mean %.4f"
              % (label, (H > 0.5).sum(), 100 * (H > 0.5).mean(), (H > 0.9).sum(), H.mean()))

    scene.export(args.out)
    print("\nSaved %s   (left -> right: %s)"
          % (args.out, ", ".join(lbl for lbl, _ in entries)))


if __name__ == "__main__":
    main()
