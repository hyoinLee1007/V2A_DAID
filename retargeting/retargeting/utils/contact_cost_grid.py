"""Precompute the attract cost as a field in the object's own frame.

The attract term asks, for a hand point ``p``,

    cost(p) = min over heatmap points x of  [ ||p - x|| + alpha*(1 - H(x)) ]

and pays for that minimum every reward evaluation: 4096 worlds x 64 hand points
x 2000 candidates is half a billion distances per step, and it is what took the
optimizer from 6.5 to 50 seconds per iteration.

None of that work needs repeating. The heatmap points never move *in the
object's frame* — the handle stays where it is on the cup — so ``cost`` is a
fixed function of position there, independent of where the object has been
carried. Sampling it once onto a grid turns the per-step cost from a minimum
over thousands of candidates into a single interpolated lookup.

    setup    : cells x candidates, once           (seconds on the GPU)
    runtime  : worlds x hand points lookups       (~2000x less work)

The transform direction flips as a consequence: hand points go into the object
frame rather than heatmap points coming out into the world. That flip is not
itself the saving — it moves 64 points instead of 2000, about 1.5% of the cost
— but it is what makes the lookup possible at all.

Outside the grid the value is extrapolated as ``G(clamped) + ||q - clamped||``.
By the triangle inequality that is an upper bound on the true cost, exact
whenever the nearest heatmap point lies beyond the clamp — which is the case
for a hand approaching from outside. It stays monotonic in distance, so the
term keeps pulling in the right direction out there.
"""

from __future__ import annotations

import numpy as np
import torch

# Cells evaluated per setup batch. Bounds peak memory during construction; the
# result is identical whatever this is.
_BUILD_BATCH = 65536


def build_cost_grid(verts_local: torch.Tensor, conf: torch.Tensor, alpha: float,
                    resolution: int = 256, margin: float = 0.20):
    """Sample ``min_x [||q - x|| + alpha*(1 - H(x))]`` over a grid.

    ``verts_local`` are the heatmap points in the object's frame, so the grid
    is too, and it stays valid however the object moves.

    ``margin`` extends the grid past the object's bounding box. It trades reach
    against resolution at fixed memory: the term only needs precision where the
    hand is near the object, and the extrapolation above covers the rest.

    Returns ``(grid, lo, hi)`` with grid shaped ``(D, H, W)``.
    """
    dev = verts_local.device
    lo = verts_local.min(0).values - margin
    hi = verts_local.max(0).values + margin

    axes = [torch.linspace(float(lo[i]), float(hi[i]), resolution, device=dev)
            for i in range(3)]
    # 'ij' indexing keeps axis order (x, y, z) -> (D, H, W), which is what the
    # lookup below assumes when it reverses the coordinates for grid_sample.
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    w = alpha * (1.0 - conf)                                   # (V,)
    out = torch.empty(pts.shape[0], device=dev, dtype=torch.float32)
    for s in range(0, pts.shape[0], _BUILD_BATCH):
        q = pts[s:s + _BUILD_BATCH]
        d = torch.cdist(q, verts_local)                        # (b, V)
        out[s:s + _BUILD_BATCH] = (d + w[None, :]).min(dim=1).values
    return out.reshape(resolution, resolution, resolution), lo, hi


def lookup(grid: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
           q: torch.Tensor) -> torch.Tensor:
    """Trilinearly interpolate the grid at object-frame points ``q`` (..., 3)."""
    shape = q.shape[:-1]
    flat = q.reshape(-1, 3)

    clamped = torch.max(torch.min(flat, hi), lo)
    outside = torch.linalg.norm(flat - clamped, dim=-1)

    # grid_sample wants normalised coords in [-1, 1] ordered (x, y, z) against
    # tensor dims (W, H, D) — the reverse of the grid's own axis order, so the
    # coordinates are flipped here. Getting this backwards silently transposes
    # the field, which is why build/lookup are validated against the exact
    # minimum rather than trusted.
    norm = 2.0 * (clamped - lo) / (hi - lo) - 1.0
    g = norm.flip(-1).reshape(1, 1, 1, -1, 3)
    val = torch.nn.functional.grid_sample(
        grid[None, None], g, mode="bilinear", align_corners=True
    ).reshape(-1)
    return (val + outside).reshape(shape)


def exact(verts_local: torch.Tensor, conf: torch.Tensor, alpha: float,
          q: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """The minimum computed directly, for validating the grid against."""
    shape = q.shape[:-1]
    flat = q.reshape(-1, 3)
    w = alpha * (1.0 - conf)
    best = None
    for s in range(0, verts_local.shape[0], chunk):
        d = torch.cdist(flat, verts_local[s:s + chunk]) + w[None, s:s + chunk]
        m = d.min(dim=1).values
        best = m if best is None else torch.minimum(best, m)
    return best.reshape(shape)
