"""Export a contact_heatmap.npz (from manual_contact_heatmap.py) as a mouse-navigable .glb.

Colors the object mesh by contact_confidence (matplotlib colormap) and marks the
hand-picked seed vertices with small red spheres, so it's clear where the
heatmap was anchored vs. how far it spread (contact_confidence decays with
geodesic distance from the seeds).

Usage (any env with trimesh + matplotlib, e.g. `retargeting` conda env):
    python visualize_contact_heatmap_glb.py --npz heatmap_out/contact_heatmap.npz

Open the resulting .glb in any glTF viewer (Blender, VS Code's glTF preview,
or e.g. https://gltf-viewer.donmccurdy.com/) and orbit with the mouse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="heatmap_out/contact_heatmap.npz")
    p.add_argument("--out", default=None)
    p.add_argument("--colormap", default="viridis", help="Any matplotlib colormap name.")
    p.add_argument("--seed-radius-frac", type=float, default=0.01,
                    help="Seed marker sphere radius as a fraction of the mesh bbox diagonal.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    d = np.load(str(npz_path))

    vertices = d["vertices"]
    faces = d["faces"]
    contact_confidence = d["contact_confidence"]       # (V,) in [0, 1]
    seed_vertex_indices = d["seed_vertex_indices"]
    bbox_diagonal = float(d["bbox_diagonal"])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    cmap = matplotlib.colormaps[args.colormap]
    colors = (cmap(np.clip(contact_confidence, 0.0, 1.0))[:, :4] * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="object")

    seed_radius = args.seed_radius_frac * bbox_diagonal
    for i, vi in enumerate(seed_vertex_indices):
        s = trimesh.creation.icosphere(radius=seed_radius, subdivisions=1)
        s.apply_translation(vertices[int(vi)])
        s.visual = trimesh.visual.ColorVisuals(
            mesh=s, vertex_colors=np.tile([255, 0, 0, 255], (len(s.vertices), 1))
        )
        scene.add_geometry(s, geom_name=f"seed_{i}")

    out_path = Path(args.out) if args.out else npz_path.with_name(npz_path.stem + "_viz.glb")
    scene.export(str(out_path))
    print(f"Saved {out_path}  ({len(vertices)} verts, {len(seed_vertex_indices)} seeds, "
          f"confidence range [{contact_confidence.min():.3f}, {contact_confidence.max():.3f}])")


if __name__ == "__main__":
    main()
