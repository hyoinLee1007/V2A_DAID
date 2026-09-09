"""Rebuild contact_heatmap.npz in visual.obj vertex order.

Finding (2026-08-26): contact_heatmap.npz's vertices/faces are in the SAM-3D
mesh's own vertex ORDER, which does NOT match the retargeting pipeline's
outputs/assets/objects/cupmove/visual.obj order (faces 100% differ; no
identity correspondence even allowing a global rigid+scale transform). Any
consumer that pairs npz `contact_confidence[i]` with visual-mesh vertex i
(contact_region_reward.py, viz_backoff_compare.py) was therefore reading
confidence at effectively random vertices.

This script recovers the vertex permutation geometrically (same shape, same
vertex count -> nearest-neighbor bijection after aligning frames with
PCA-init + Kabsch/ICP refinement) and writes a corrected npz:

    vertices           visual.obj positions (obj order, obj scale/frame)
    faces              visual.obj faces
    contact_confidence npz confidence remapped to obj order
    (other per-vertex arrays remapped the same way; scalars copied)

Run:
    conda run -n retargeting python fix_heatmap_vertex_order.py \
        --obj  ../retargeting/outputs/assets/objects/cupmove/visual.obj \
        --npz  heatmap_out/contact_heatmap.npz \
        --out  heatmap_out/contact_heatmap_objorder.npz
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj", default="../retargeting/outputs/assets/objects/cupmove/visual.obj")
    p.add_argument("--npz", default="heatmap_out/contact_heatmap.npz")
    p.add_argument("--out", default="heatmap_out/contact_heatmap_objorder.npz")
    return p.parse_args()


def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation R minimizing ||dst - src @ R.T|| for paired points."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def main() -> None:
    args = parse_args()
    obj_v, obj_f = load_obj(args.obj)
    data = np.load(args.npz)
    npz_v = data["vertices"].astype(np.float64)
    N = obj_v.shape[0]
    assert npz_v.shape[0] == N, f"vertex count mismatch: {npz_v.shape[0]} vs {N}"

    # Normalize both point sets: center + unit RMS radius (removes the
    # SAM-3D-vs-processed global scale difference, ~7x here).
    oc = obj_v - obj_v.mean(0)
    hc = npz_v - npz_v.mean(0)
    oc /= np.sqrt((oc**2).sum(1).mean())
    hc /= np.sqrt((hc**2).sum(1).mean())

    # PCA axes as rotation init. Principal directions match up to axis sign
    # flips; try all 4 proper-rotation sign combinations and keep the one
    # with the lowest nearest-neighbor cost.
    def pca_axes(x: np.ndarray) -> np.ndarray:
        _, _, Vt = np.linalg.svd(x, full_matrices=False)
        V = Vt.T
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1
        return V

    Vo, Vh = pca_axes(oc), pca_axes(hc)
    tree_h = cKDTree(hc)
    best = None
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        S = np.diag([sx, sy, sx * sy])  # keeps det=+1
        R0 = Vh @ S @ Vo.T
        dd, _ = tree_h.query(oc @ R0.T)
        cost = float(dd.mean())
        if best is None or cost < best[0]:
            best = (cost, R0)
    cost, R = best
    print(f"PCA init: mean NN dist = {cost:.6f} (unit-RMS scale)")

    # ICP refinement: alternate NN correspondence and Kabsch.
    for it in range(10):
        dd, ii = tree_h.query(oc @ R.T)
        R = kabsch(oc, hc[ii])
        print(f"  icp iter {it}: mean NN dist = {dd.mean():.8f}")
        if dd.mean() < 1e-9:
            break

    dd, perm = tree_h.query(oc @ R.T)
    n_unique = len(np.unique(perm))
    print(f"final: mean={dd.mean():.8f} max={dd.max():.8f} unique targets={n_unique}/{N}")
    # Not necessarily a strict permutation: both meshes carry coincident
    # duplicate vertices (UV-seam twins sharing one 3D position), so several
    # obj verts may map to the same npz vert. That is fine for transferring
    # per-vertex values — what matters is that every obj vertex found an
    # npz vertex at (numerically) the SAME position.
    if dd.max() > 1e-3:
        raise SystemExit(
            "Nearest-neighbor distances too large — the meshes do not "
            "coincide geometrically; aborting rather than writing a wrong "
            "mapping."
        )

    per_vertex = {}
    scalars = {}
    for k in data.files:
        arr = data[k]
        if arr.ndim >= 1 and arr.shape[0] == N and k not in ("vertices", "faces"):
            per_vertex[k] = arr[perm]
        elif k in ("vertices", "faces"):
            continue
        else:
            scalars[k] = arr

    # seed_vertex_indices index INTO the npz order; translate to obj order by
    # a reverse position query (perm may be non-injective, so it has no
    # proper inverse).
    if "seed_vertex_indices" in scalars:
        tree_o = cKDTree(oc)
        seeds = scalars.pop("seed_vertex_indices").astype(np.int64)
        _, obj_seed_idx = tree_o.query(hc[seeds] @ R)
        scalars["seed_vertex_indices"] = obj_seed_idx.astype(np.int64)

    np.savez(
        args.out,
        vertices=obj_v,
        faces=obj_f,
        **per_vertex,
        **scalars,
    )
    conf = per_vertex["contact_confidence"]
    high = conf > 0.5
    ext = np.ptp(obj_v[high], axis=0) if high.sum() else None
    print(f"saved {args.out}")
    print(f"sanity: conf>0.5 verts={int(high.sum())}, their obj-frame bbox extent={ext}")
    print("        (a real handle region should be small & one-sided, not cup-sized)")


if __name__ == "__main__":
    main()
