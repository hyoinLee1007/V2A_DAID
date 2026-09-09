"""Send each skin point to *its own* demonstrated target, not the nearest one.

``robot_contact_reward.py``'s attract term asks, for every mapped skin point,
"how far is the nearest high-confidence object vertex?". That question has the
same answer for two fingers standing next to each other, so nothing stops them
from piling onto one spot. Measured on cupmove: the index and middle fingers
sat 8.81 cm and 7.19 cm from "their" target — the same target — while the
demonstration threads them through opposite sides of the handle loop. No run
has ever threaded it.

The anchor that produced the map was a *pair*: MANO vertex 138 met object
vertex 4021, not "object vertex 4021 was touched by something".
``extract_contact_map.py`` splatted the two ends into two separate 1-D
histograms and the pairing died there. It now saves the pairs, and this module
uses them:

    c_k = min over v in targets(k) of  ||p_k - x_v|| + alpha * (1 - w_kv)

``targets(k)`` is the set of object vertices that hand point k actually met,
and ``w_kv`` is that pair's share of k's anchors, so a point is pulled hardest
towards where it spent most of the demonstration.

Two things follow for free:

- **No grid.** The grid existed because the unpaired minimum ran over 2000
  candidates every step. Here the median point has 22 targets, so the exact
  minimum is 64 x 58 at worst — cheaper than the lookup it replaces.
- **A weighted mean, not a sum.** The old term summed over 64 points, and
  since the weights add to 42.13 that multiplied the whole term by 42 against
  everything else in the reward. Measured on run E: attract 8.53 against
  qpos_rew 0.29, so abandoning the reference cost almost nothing and the
  optimizer duly abandoned it. Dividing by the weight sum makes the scale mean
  "reward per metre of mean point-to-target distance", which is a number that
  can be set from a measurement.
"""

from __future__ import annotations

import numpy as np
import torch


def build_targets(tgt_offset: np.ndarray, tgt_vertex: np.ndarray,
                  tgt_weight: np.ndarray, keep: np.ndarray,
                  local_verts: np.ndarray, alpha: float,
                  device, dtype=torch.float32
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Padded per-point targets for the kept points, in the object's frame.

    Returns ``(xyz, offset, empty)`` of shapes ``(K, T, 3)``, ``(K, T)`` and
    ``(K,)``, where
    ``offset`` already carries ``alpha * (1 - w_hat)`` and is set to a large
    value on the padding, so a plain ``min`` over T ignores it without needing
    a mask. ``None`` if the map carries no pairs.
    """
    if tgt_vertex is None or len(tgt_vertex) == 0:
        return None

    lists_v, lists_w = [], []
    for k in keep:
        s, e = int(tgt_offset[k]), int(tgt_offset[k + 1])
        lists_v.append(tgt_vertex[s:e])
        lists_w.append(tgt_weight[s:e])
    width = max((len(v) for v in lists_v), default=0)
    if width == 0:
        return None

    K = len(lists_v)
    xyz = np.zeros((K, width, 3), np.float64)
    off = np.full((K, width), 1e3, np.float64)      # padding: never the minimum
    for i, (v, w) in enumerate(zip(lists_v, lists_w)):
        if len(v) == 0:
            # A point with no pair of its own would otherwise be pulled
            # nowhere; leaving the row entirely padded makes its cost the
            # constant 1e3, which would swamp everything. Give it a zero-cost
            # row at its own current position instead — it simply stops
            # contributing rather than dominating.
            continue
        xyz[i, :len(v)] = local_verts[v]
        off[i, :len(v)] = alpha * (1.0 - w / max(w.max(), 1e-12))

    empty = np.array([len(v) == 0 for v in lists_v])
    return (torch.tensor(xyz, dtype=dtype, device=device),
            torch.tensor(off, dtype=dtype, device=device),
            torch.tensor(empty, device=device))


def paired_cost(pts_local: torch.Tensor, tgt_xyz: torch.Tensor,
                tgt_off: torch.Tensor) -> torch.Tensor:
    """``(W, K)`` minimum cost of each point over its own targets.

    ``pts_local`` is ``(W, K, 3)`` in the object's frame. The distance is
    expanded as ``|p|^2 - 2 p.x + |x|^2`` so the ``(W, K, T, 3)`` difference is
    never materialised — at 4096 worlds that tensor alone would be 180 MB.
    """
    pp = (pts_local * pts_local).sum(-1, keepdim=True)          # (W, K, 1)
    xx = (tgt_xyz * tgt_xyz).sum(-1)[None]                      # (1, K, T)
    px = torch.einsum("wkd,ktd->wkt", pts_local, tgt_xyz)       # (W, K, T)
    d2 = (pp + xx - 2.0 * px).clamp_min(0.0)
    return (d2.sqrt() + tgt_off[None]).amin(dim=-1)             # (W, K)


def weighted_mean(cost: torch.Tensor, weight: torch.Tensor,
                  empty: torch.Tensor) -> torch.Tensor:
    """``sum_k w_k c_k / sum_k w_k`` over the points that have targets.

    A mean, not a sum. The sum is what made the term 42x larger than every
    other term in the reward for no reason anyone chose.
    """
    w = torch.where(empty, torch.zeros_like(weight), weight)
    denom = w.sum().clamp_min(1e-12)
    return (cost * w[None]).sum(-1) / denom
