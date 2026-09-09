"""Reward the demonstrated *patches of skin*, not just proximity to the object.

``contact_region_reward.py`` scores five fingertip sites against an object-side
heatmap. That cannot express what the cupmove demonstration actually does. Two
gaps, both measured on this task:

- **It only sees the fingertips.** 55% of the demonstrated contact weight sits
  on proximal and middle phalanges and the palm — links with no site at all.
  The hand hooks through the mug handle; it does not pinch it.
- **It cannot tell pad from back.** Threading a finger through the handle puts
  the pad and the back of that finger against the same ring of object
  vertices, so an object-only map scores the demonstrated grasp and the
  back-of-hand grasp identically. This is the failure the reward kept
  producing.

Both are fixed by scoring the robot's own skin. ``build_robot_contact_map.py``
carries the hand half of the CHOIR anchors through the MANO->robot
correspondence, giving points ``p_k`` on named links with weights ``w_k``.
Their world positions come from the same expression MuJoCo uses for a site,

    p_k = xpos[body_k] + xmat[body_k] @ offset_k

read straight off ``data_wp``, so no site is added to the robot XML.

The term has two halves and needs both:

    attract = sum_k w_k * min_x [ ||p_k - x||^2 + alpha*(1 - H(x)) ]
    repel   = sum over hand<->object contacts of  max(0, dist_to_nearest_p_k - r)

    reward -= attract_scale * attract + repel_scale * repel

A penalty on wrong-surface contact by itself does not work, and this is not a
guess — it was tried, three times, as ``palm_side_contact.py``. With only a
penalty the cheapest solution is to stop touching the object at all, and that
is what the optimizer found. The attraction term removes that escape: letting
go forfeits it. The earlier attempt also had no attractor to fall back on and
was ~8000x smaller than the heatmap term it competed with.

Finger identity comes along for free. Each ``p_k`` sits on a specific link, so
"the index finger's middle phalanx touches the handle" is expressible without
any explicit per-finger assignment.
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import warp as wp

from retargeting.config import Config
from retargeting.utils.contact_cost_grid import lookup as _grid_lookup
from retargeting.utils.contact_region_reward import (
    _rotate_points_by_quat, _visual_mesh_local_verts,
)

# Heatmap points compared per chunk. The pairwise tensor is
# worlds x K x chunk; chunking keeps that bounded no matter how many
# candidate points the heatmap has.
_CHUNK = 256


def _setup(config: Config, env) -> None:
    """One-time load of the robot contact map. Cached on ``config``.

    Any failure leaves the caches None so the term degrades to a no-op with a
    single warning, matching contact_region_reward.py — this must never be able
    to crash the optimizer.
    """
    if hasattr(config, "_robot_contact_ready"):
        return
    config._robot_contact_ready = True
    config._robot_contact_body = None
    config._robot_contact_offset = None
    config._robot_contact_weight = None
    config._robot_contact_verts = None
    config._robot_contact_conf = None
    config._robot_contact_grid = None

    if not config.robot_contact_map_path:
        loguru.logger.warning(
            "robot contact reward is on but robot_contact_map_path is empty; "
            "term will be a no-op."
        )
        return

    try:
        d = np.load(config.robot_contact_map_path, allow_pickle=True)
    except OSError as exc:
        loguru.logger.warning("Could not read {}: {}; term will be a no-op.",
                              config.robot_contact_map_path, exc)
        return

    body = np.asarray(d["body_id"], dtype=np.int64)
    offset = np.asarray(d["local_offset"], dtype=np.float64)
    weight = np.asarray(d["weight"], dtype=np.float64)
    if len(body) == 0:
        loguru.logger.warning("Robot contact map is empty; term will be a no-op.")
        return

    model = env.model_cpu
    if body.max() >= model.nbody:
        loguru.logger.warning(
            "Robot contact map references body {} but the scene has {}; it was "
            "built against a different scene. Term will be a no-op.",
            int(body.max()), model.nbody)
        return

    # The map stores body *ids*, which are only meaningful against the scene it
    # was built from. Body order happens to be stable across scenes generated
    # for the same robot, but a silently shifted id would put every contact
    # target on the wrong link while looking perfectly healthy, so check the
    # names the map recorded alongside them.
    if "body_name" in d.files:
        import mujoco
        want = d["body_name"].astype(str)
        got = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i))
               for i in body]
        bad = [(int(i), w, g) for i, w, g in zip(body, want, got) if w != g]
        if bad:
            loguru.logger.warning(
                "Robot contact map body ids do not match this scene "
                "({} of {} differ, e.g. id {} is {!r} here but {!r} in the map). "
                "Rebuild it against this scene. Term will be a no-op.",
                len(bad), len(body), bad[0][0], bad[0][2], bad[0][1])
            return

    # Keep the heaviest points. The pairwise cost is linear in this count and
    # the light tail contributes almost nothing to a weighted sum.
    cap = int(config.robot_contact_max_points)
    keep = np.arange(len(weight))
    if 0 < cap < len(weight):
        keep = np.argpartition(-weight, cap)[:cap]
        body, offset, weight = body[keep], offset[keep], weight[keep]

    side = config.embodiment_type
    local_verts = _visual_mesh_local_verts(model, side)
    if local_verts is None:
        loguru.logger.warning(
            "No '{}_visual' mesh geom; robot contact reward will be a no-op.", side)
        return

    heat = np.load(config.contact_heatmap_path)
    conf = heat["contact_confidence"].astype(np.float64)
    if conf.shape[0] != local_verts.shape[0]:
        loguru.logger.warning(
            "heatmap has {} vertices but {}_visual has {}; robot contact reward "
            "will be a no-op.", conf.shape[0], side, local_verts.shape[0])
        return

    max_pts = int(config.contact_region_max_points)
    sel = (np.argpartition(-conf, max_pts)[:max_pts]
           if 0 < max_pts < conf.shape[0] else np.arange(conf.shape[0]))

    dev = config.device
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)  # noqa: E731
    config._robot_contact_body = t(body, torch.long)
    config._robot_contact_offset = t(offset)
    config._robot_contact_weight = t(weight)
    config._robot_contact_verts = t(local_verts[sel])
    config._robot_contact_conf = t(conf[sel])

    # Per-point target lists, when the map carries the anchor pairs. This path
    # replaces the global-nearest minimum (and with it the grid), so build it
    # before the grid and skip the grid if it succeeds.
    config._robot_contact_tgt = None
    if config.robot_contact_paired and "tgt_vertex" in d.files:
        from retargeting.utils.paired_contact_cost import build_targets
        built = build_targets(
            np.asarray(d["tgt_offset"], np.int64),
            np.asarray(d["tgt_vertex"], np.int64),
            np.asarray(d["tgt_weight"], np.float64),
            keep, local_verts, float(config.robot_contact_alpha), dev)
        if built is not None:
            config._robot_contact_tgt = built
            n_t = int((built[1] < 1e2).sum())
            loguru.logger.info(
                "Robot contact reward: paired targets on, {} pairs over {} "
                "points ({} without a target of their own); the global-nearest "
                "minimum and its grid are bypassed.",
                n_t, built[0].shape[0], int(built[2].sum()))
    elif config.robot_contact_paired:
        loguru.logger.warning(
            "robot_contact_paired is on but {} has no tgt_vertex; rebuild it "
            "with build_robot_contact_map.py. Falling back to global-nearest.",
            config.robot_contact_map_path)

    # Palmar axis of every hand link, in that link's own frame so it stays
    # valid as the hand moves, plus the offset of the link's mid-surface along
    # it. Read at the open pose, where the segment frames are defined.
    from retargeting.utils.mano_robot_map import (
        _bid, _canonical, _palmar_world, _surface_world, segment_frame, segments)
    cd = _canonical(model, side)
    ax = np.zeros((model.nbody, 3))
    ctr = np.zeros(model.nbody)
    is_link = np.zeros(model.nbody, bool)
    for seg in segments(side):
        fb = _bid(model, seg["rob_frame"])
        fr = segment_frame(model, cd, seg["rob_frame"], seg["rob_distal"],
                           _palmar_world(model, side, seg["finger"]),
                           _surface_world(model, cd, seg["rob_surf"]))
        z_local = cd.xmat[fb].reshape(3, 3).T @ fr["z"]
        for nm in list(seg["rob_surf"]) + [seg["rob_frame"]]:
            b = _bid(model, nm)
            ax[b], ctr[b], is_link[b] = z_local, fr["cz"], True
    config._robot_contact_palmar_axis = t(ax)
    config._robot_contact_palmar_ctr = t(ctr)
    config._robot_contact_is_link = torch.tensor(is_link, device=dev)
    config._robot_contact_geom_body = torch.tensor(
        np.asarray(model.geom_bodyid, dtype=np.int64), dtype=torch.long, device=dev)

    # Precompute the attract cost as a field in the object's frame. Validated
    # against the exact minimum at 0.005 mm mean error, and ~100x faster —
    # 83 ms per step of pairwise minima becomes a 0.8 ms lookup.
    config._robot_contact_grid = None
    config._robot_contact_grid_lo = None
    config._robot_contact_grid_hi = None
    if config.robot_contact_grid_res > 0 and config._robot_contact_tgt is None:
        from retargeting.utils.contact_cost_grid import build_cost_grid
        g, glo, ghi = build_cost_grid(
            config._robot_contact_verts, config._robot_contact_conf,
            config.robot_contact_alpha,
            resolution=int(config.robot_contact_grid_res),
            margin=float(config.robot_contact_grid_margin))
        config._robot_contact_grid = g
        config._robot_contact_grid_lo = glo
        config._robot_contact_grid_hi = ghi
        loguru.logger.info(
            "Robot contact reward: cost grid {}^3 ({:.0f} MB) over the object's "
            "frame, margin {:.2f} m", int(config.robot_contact_grid_res),
            g.numel() * 4 / 1e6, config.robot_contact_grid_margin)

    share = ""
    if "finger" in d.files:
        uniq, cnt = np.unique(d["finger"].astype(str)[keep], return_counts=True)
        share = "  (" + " ".join(f"{u}:{c}" for u, c in zip(uniq, cnt)) + ")"
    loguru.logger.info(
        "Robot contact reward: {} skin points over {} heatmap points{}",
        len(body), len(sel), share)


def _contact_points_world(config: Config, env) -> torch.Tensor:
    """``xpos[body] + xmat[body] @ offset`` for every mapped point, all worlds."""
    xpos = wp.to_torch(env.data_wp.xpos)                       # (W, nbody, 3)
    xmat = wp.to_torch(env.data_wp.xmat).reshape(xpos.shape[0], -1, 3, 3)
    b = config._robot_contact_body
    return xpos[:, b] + torch.einsum("wnij,nj->wni", xmat[:, b],
                                     config._robot_contact_offset)


def compute_robot_contact_terms(
    config: Config, env, qpos_sim: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(attract, repel)`` per world, each already scaled. Zeros if set-up failed."""
    _setup(config, env)
    zero = torch.zeros(qpos_sim.shape[0], device=qpos_sim.device)
    if config._robot_contact_body is None:
        return zero, zero

    pts = _contact_points_world(config, env)                   # (W, K, 3)

    attract = zero
    if config.robot_contact_attract_scale > 0.0:
        obj_pos, obj_quat = qpos_sim[:, -7:-4], qpos_sim[:, -4:]

        if config._robot_contact_tgt is not None:
            # Each point against its own demonstrated targets, and a weighted
            # mean rather than a sum — see paired_contact_cost.py for why both
            # of those are the fix and not a tuning knob.
            from retargeting.utils.paired_contact_cost import (
                paired_cost, weighted_mean)
            tgt_xyz, tgt_off, tgt_empty = config._robot_contact_tgt
            q_inv = obj_quat.clone()
            q_inv[:, 1:] = -q_inv[:, 1:]
            local = _rotate_points_by_quat(q_inv[:, None, :],
                                           pts - obj_pos[:, None, :])
            cost = paired_cost(local, tgt_xyz, tgt_off)        # (W, K)
            attract = config.robot_contact_attract_scale * weighted_mean(
                cost, config._robot_contact_weight, tgt_empty)
            return attract, (_repel(config, env, pts)
                             if config.robot_contact_repel_scale > 0.0 else zero)

        if config._robot_contact_grid is not None:
            # The cost is a fixed field in the object's frame, so bring the
            # hand to it and read the answer off. Going the other way — the
            # heatmap out to the world — is what forces the minimum to be
            # recomputed from scratch every step, since the candidates land
            # somewhere new each time.
            q_inv = obj_quat.clone()
            q_inv[:, 1:] = -q_inv[:, 1:]
            local = _rotate_points_by_quat(q_inv[:, None, :],
                                           pts - obj_pos[:, None, :])
            best = _grid_lookup(
                config._robot_contact_grid, config._robot_contact_grid_lo,
                config._robot_contact_grid_hi, local)          # (W, K)
            attract = config.robot_contact_attract_scale * (
                best * config._robot_contact_weight[None, :]).sum(-1)
            return attract, (_repel(config, env, pts)
                             if config.robot_contact_repel_scale > 0.0 else zero)

        verts = (_rotate_points_by_quat(obj_quat[:, None, :],
                                        config._robot_contact_verts[None])
                 + obj_pos[:, None, :])                        # (W, V, 3)

        # Distance, not squared distance. Squared is negligible at the scale
        # that matters here: 2 cm gives 4e-4 against an alpha offset of 1e-2,
        # so "touching" and "2 cm off" score within 4% of each other and the
        # term cannot hold a grasp. Measured — letting go of the object cost
        # only +0.29 against a 3.5 deterrent, and the optimizer let go. Linear
        # makes that same 2 cm worth 0.02, fifty times more signal, and matches
        # the form qpos_rew already uses (weights inside an L2 norm).
        best = None
        for s in range(0, verts.shape[1], _CHUNK):
            vc = verts[:, s:s + _CHUNK]                        # (W, c, 3)
            d = torch.linalg.norm(pts[:, :, None, :] - vc[:, None, :, :], dim=-1) # TODO : 손(64개 후보점) - Heatmap점(256)간의 모든 쌍 거리 생성
            cost = d + config.robot_contact_alpha * ( # TODO : d + alpha(1-H) => 가깝고 + 신뢰도 높은 정보를 점수화
                1.0 - config._robot_contact_conf[None, None, s:s + _CHUNK])
            chunk_min = cost.min(dim=-1).values                # (W, K)
            best = chunk_min if best is None else torch.minimum(best, chunk_min)
        attract = config.robot_contact_attract_scale * (
            best * config._robot_contact_weight[None, :]).sum(-1)

    repel = zero
    if config.robot_contact_repel_scale > 0.0:
        repel = _repel(config, env, pts)

    return attract, repel


def _repel(config: Config, env, pts: torch.Tensor) -> torch.Tensor:
    """Penalize contact on the back of a finger.

    The contact is written in its link's own frame and read along the palmar
    axis; negative is the back. A sign test, unambiguous however thin the link.

    An earlier version also penalized contacts far from any mapped point, for
    links the demonstration never used. That branch was dead: every body on
    this hand that carries geometry is mapped, and the six that are not
    (``*_base_tx`` and friends) are massless spacers with no geom at all, so
    nothing can touch the object through them.

    Distance alone was tried first and never fired once. A phalanx is about
    15 mm thick, so a contact on the *back* of the middle finger is barely a
    centimetre from the demonstrated point on its pad — comfortably inside any
    free radius wide enough to be useful. Measured on that run: every contact,
    including the 156 dorsal ones, sat within 2 cm of a demonstrated point,
    mean 0.91 cm. Euclidean distance cannot separate the two faces of a thin
    finger, and the sign of the palmar coordinate is exactly the quantity that
    can.
    """
    
    # TODO : MuJoCo의 접촉 목록에는 Scene의 모든 충돌이 들어있음 -> 손<->물체 접촉만 골라내기
    # Populated by mjwp's contact set-up; absent if this runs before it.
    if not hasattr(config, "_is_hand_geom"):
        return torch.zeros(pts.shape[0], device=pts.device)
    nacon = int(wp.to_torch(env.data_wp.nacon).item())
    if nacon == 0:
        return torch.zeros(pts.shape[0], device=pts.device)

    dist = wp.to_torch(env.data_wp.contact.dist).reshape(-1)[:nacon]
    cpos = wp.to_torch(env.data_wp.contact.pos).reshape(-1, 3)[:nacon]
    geom = wp.to_torch(env.data_wp.contact.geom).reshape(-1, 2)[:nacon]
    wid = (wp.to_torch(env.data_wp.contact.worldid).reshape(-1)[:nacon]
           .long().clamp(0, pts.shape[0] - 1))

    ngeom = config._is_hand_geom.shape[0]
    g1 = geom[:, 0].long().clamp(0, ngeom - 1)
    g2 = geom[:, 1].long().clamp(0, ngeom - 1)



    # TODO : hand<->object인지 확인 
    hand_obj = ((config._is_hand_geom[g1] & config._is_object_geom[g2])
                | (config._is_object_geom[g1] & config._is_hand_geom[g2]))

    touching = (torch.nan_to_num(dist, nan=1.0) <= 0.0) & hand_obj
    if not bool(touching.any()):
        return torch.zeros(pts.shape[0], device=pts.device)

    idx = touching.nonzero(as_tuple=True)[0]
    w_idx = wid[idx]
    # TODO : What's the link
    hand_geom = torch.where(config._is_hand_geom[g1[idx]], g1[idx], g2[idx])
    b = config._robot_contact_geom_body[hand_geom]
    on_link = config._robot_contact_is_link[b]

    # Which face: the contact in its link's frame, along that link's palmar
    # axis. Negative is the back of the finger.
    xpos = wp.to_torch(env.data_wp.xpos)
    xmat = wp.to_torch(env.data_wp.xmat).reshape(xpos.shape[0], -1, 3, 3)

    # TODO : 앞면인지 뒷면인지 확인
    rel = cpos[idx] - xpos[w_idx, b] # cpos(접촉위치), xpos(링크원점)
    local = torch.einsum("nji,nj->ni", xmat[w_idx, b], rel)   # R^T @ rel => Link좌표계로 회전 => 손가락이 어떻게 굽어도 항상 뼈축과 같이 움직이니까 OK
    palmar = ((local * config._robot_contact_palmar_axis[b]).sum(-1)
              - config._robot_contact_palmar_ctr[b])
    # ``on_link`` zeroes anything without a palmar axis rather than trusting
    # the zero vector such a link would carry.
    bad = torch.where(on_link, torch.clamp(-palmar, min=0.0),
                      torch.zeros_like(palmar))

    out = torch.zeros(pts.shape[0], device=pts.device)
    out.scatter_add_(0, w_idx, bad)
    return config.robot_contact_repel_scale * out
