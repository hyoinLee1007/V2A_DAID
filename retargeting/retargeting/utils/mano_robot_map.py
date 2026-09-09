"""Correspondence from MANO surface vertices to points on the robot hand.

CHOIR's anchors are pairs ``(MANO vertex id, object face id + barycentric)``.
The object half transfers to this pipeline unchanged — the object mesh is the
same mesh. The hand half does not: MANO vertex 138 is an index into a
778-vertex human mesh and means nothing to a sharpa hand, which has a different
topology, different proportions and 41 rigid bodies.

This module builds the missing half: for every MANO vertex it produces

    (robot body id, offset expressed in that body's own frame)

so the corresponding point on the robot at any pose is the same expression
MuJoCo uses internally for a site,

    p_world = xpos[body] + xmat[body] @ local_offset

evaluated on GPU from ``data_wp.xpos`` / ``data_wp.xmat``. Nothing is added to
the robot XML: a site *is* a (body, offset) pair with a name, so naming it
buys nothing that this table does not already carry. It also keeps the scene
generator untouched — ``resolve_pedestal`` pairs every ``track_*`` site with a
``ref_*`` one and would fail to compile on an unpaired new site.

How the correspondence is derived
---------------------------------
Both hands are MJCF models, so both go through the identical procedure.

1. **Segment.** Each MANO vertex is assigned to the phalanx it is skinned to
   (``argmax`` of the MANO skinning weights). The MANO MJCF in
   ``assets/robots/mano`` already carries exactly that split as per-part
   meshes — its part vertices reproduce ``v_template - J[0]`` to under a
   micrometre — so the segmentation is read off the asset and needs no
   skinning weights at runtime.

2. **Frame.** Every segment gets an orthonormal frame built from its own
   kinematics, on both hands:

       origin  the proximal joint (the body's frame origin on both models)
       x       the bone axis, toward the distal joint (toward the tip site
               for a distal phalanx)
       z       the palmar direction, measured by flexing the finger and
               watching which way the tip moves, then orthogonalised to x
       y       z cross x

   The palmar probe drives only the *flexion* joints. They are told from the
   abduction joints by the asymmetry of their range — flexion runs one way
   (``[0, 1.745]``, ``[-0.175, 1.571]``) while abduction is symmetric
   (``[-0.349, 0.349]``). That test holds on both models without hard-coding
   joint names, which differ (``left_j_index1z`` vs ``left_index_MCP_FE``).

3. **Normalise and transfer.** A vertex is written in its segment's frame and
   divided by that segment's own extents — bone length along x, robust
   half-width along y and z, with the y/z centres subtracted so a segment
   whose bulk sits off the bone axis (the palm) still maps centre to centre.
   The same normalised triple is then multiplied by the *robot* segment's
   extents. Proportions differ between a human hand and this robot, so
   absolute offsets would not transfer; relative position within a phalanx
   does.

4. **Snap.** The transferred point rarely lands exactly on the robot's skin —
   the two shapes differ by more than three scale numbers can absorb, by about
   1.3 mm on a finger and 7 mm on the palm — so it is cast onto the visual
   mesh along the palmar normal. Travelling that way changes only the point's
   depth and leaves where it sits *on* the segment alone, which is the part
   the correspondence is about. Snapping to the nearest vertex instead slides
   it along the skin (4.3 mm on the palm) and lets neighbouring points collapse
   onto the same feature.

The visual meshes are used, not the collision ones: the collision proxies on
this robot are 17 capsules, and MANO is likewise a surface mesh, so the visual
meshes are the matching level of description.

What this does and does not give you
------------------------------------
It gives finger identity. A CHOIR anchor on a MANO thumb pad lands on the
robot's thumb pad, so "which finger touches where" survives the transfer — the
information the current reward discards when it reduces the hand to five
fingertip sites and takes a min over them.

It does not give a contact map by itself. It is the change of coordinates that
makes the hand half of a CHOIR anchor usable; what is transferred through it is
a separate question.
"""

from __future__ import annotations

import numpy as np

# Probe angle (rad) used only to read off which way flexion points.
_FLEX_PROBE = 0.3
# A joint counts as flexion when one limit dominates the other by this factor.
_ASYMMETRY = 3.0
# Half-angle of the cone that keeps a snapped point on the same side of the bone.
_SNAP_CONE_DEG = 75.0
# Percentile pair used for the robust half-extent along y and z.
_EXTENT_PCT = (2.5, 97.5)

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_LONG = ("index", "middle", "ring", "pinky")


def segments(side: str) -> list[dict]:
    """The 16 MANO segments and the robot bodies each one corresponds to.

    ``frame`` is the body whose origin and orientation define the segment's
    local frame; ``surf`` lists every body whose visual mesh forms that
    segment's skin. The two differ where the robot splits into more links than
    MANO does: ``*_MCP_VL`` is a massless spacer that exists only to give the
    2-DOF knuckle two hinges, and it carries the knuckle shell, so its mesh
    belongs to the proximal phalanx it shares an origin with.

    ``distal`` is the point the bone axis aims at, given as a body name or as
    ``("site", name)`` for the fingertips, which have no child body.
    """
    segs: list[dict] = []

    segs.append(dict(
        name="palm",
        mano_frame=f"{side}_palm", mano_surf=[f"{side}_palm"],
        mano_distal=f"{side}_middle1z",
        rob_frame=f"{side}_hand_C_MC",
        rob_surf=[f"{side}_hand_C_MC", f"{side}_pinky_MC"],
        rob_distal=f"{side}_middle_MCP_VL",
        finger="middle",
    ))

    for f in _LONG:
        segs.append(dict(
            name=f"{f}1",
            mano_frame=f"{side}_{f}1z", mano_surf=[f"{side}_{f}1z"],
            mano_distal=f"{side}_{f}2",
            rob_frame=f"{side}_{f}_PP",
            rob_surf=[f"{side}_{f}_MCP_VL", f"{side}_{f}_PP"],
            rob_distal=f"{side}_{f}_MP",
            finger=f,
        ))
        segs.append(dict(
            name=f"{f}2",
            mano_frame=f"{side}_{f}2", mano_surf=[f"{side}_{f}2"],
            mano_distal=f"{side}_{f}3",
            rob_frame=f"{side}_{f}_MP", rob_surf=[f"{side}_{f}_MP"],
            rob_distal=f"{side}_{f}_DP",
            finger=f,
        ))
        segs.append(dict(
            name=f"{f}3",
            mano_frame=f"{side}_{f}3", mano_surf=[f"{side}_{f}3"],
            mano_distal=("site", f"{side}_{f}_tip"),
            rob_frame=f"{side}_{f}_DP", rob_surf=[f"{side}_{f}_DP"],
            rob_distal=("site", f"{side}_{f}_tip"),
            finger=f,
        ))

    # The thumb has one fewer phalanx than the long fingers on both hands, so
    # its three MANO segments line up with metacarpal / proximal / distal.
    segs.append(dict(
        name="thumb1",
        mano_frame=f"{side}_thumb1z", mano_surf=[f"{side}_thumb1z"],
        mano_distal=f"{side}_thumb2y",
        rob_frame=f"{side}_thumb_MC",
        rob_surf=[f"{side}_thumb_CMC_VL", f"{side}_thumb_MC"],
        rob_distal=f"{side}_thumb_MCP_VL",
        finger="thumb",
    ))
    segs.append(dict(
        name="thumb2",
        mano_frame=f"{side}_thumb2z", mano_surf=[f"{side}_thumb2z"],
        mano_distal=f"{side}_thumb3",
        rob_frame=f"{side}_thumb_PP",
        rob_surf=[f"{side}_thumb_MCP_VL", f"{side}_thumb_PP"],
        rob_distal=f"{side}_thumb_DP",
        finger="thumb",
    ))
    segs.append(dict(
        name="thumb3",
        mano_frame=f"{side}_thumb3", mano_surf=[f"{side}_thumb3"],
        mano_distal=("site", f"{side}_thumb_tip"),
        rob_frame=f"{side}_thumb_DP", rob_surf=[f"{side}_thumb_DP"],
        rob_distal=("site", f"{side}_thumb_tip"),
        finger="thumb",
    ))
    return segs


# --------------------------------------------------------------------------
# model probing
# --------------------------------------------------------------------------


def _bid(model, name: str) -> int:
    import mujoco
    i = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if i < 0:
        raise KeyError(f"body {name!r} not in model")
    return i


def _sid(model, name: str) -> int:
    import mujoco
    i = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
    if i < 0:
        raise KeyError(f"site {name!r} not in model")
    return i


def _hand_hinges(model, side: str) -> list[int]:
    """qpos addresses of every hinge joint belonging to this hand's fingers."""
    import mujoco
    out = []
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not any(f"{side}_{f}" in nm or f"{side}_j_{f}" in nm for f in _FINGERS):
            continue
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        out.append(int(model.jnt_qposadr[j]))
    return out


def _flexion_joints(model, side: str, finger: str) -> list[tuple[int, float]]:
    """``(qposadr, probe)`` for this finger's flexion hinges only.

    Flexion is identified by an asymmetric range and signed toward the long
    side. Abduction joints are symmetric and drop out; including them tilts the
    measured direction sideways, which on this robot moved the result from
    ``[0.99, 0.00, -0.15]`` to ``[0.80, 0.51, -0.32]``.
    """
    import mujoco
    out = []
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if f"{side}_{finger}" not in nm and f"{side}_j_{finger}" not in nm:
            continue
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        lo, hi = (float(x) for x in model.jnt_range[j])
        if not bool(model.jnt_limited[j]) or hi <= lo:
            continue
        if abs(hi) < _ASYMMETRY * abs(lo) and abs(lo) < _ASYMMETRY * abs(hi):
            continue                      # symmetric -> abduction
        sign = 1.0 if abs(hi) >= abs(lo) else -1.0
        out.append((int(model.jnt_qposadr[j]),
                    float(np.clip(sign * _FLEX_PROBE, lo, hi))))
    return out


def _canonical(model, side: str):
    """MjData at the open-hand pose: rest pose with this hand's hinges zeroed."""
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for adr in _hand_hinges(model, side):
        data.qpos[adr] = 0.0
    mujoco.mj_kinematics(model, data)
    return data


def _palmar_world(model, side: str, finger: str) -> np.ndarray:
    """Direction the fingertip travels when the finger flexes, at the open pose."""
    import mujoco
    data = _canonical(model, side)
    tip = _sid(model, f"{side}_{finger}_tip")
    open_pos = data.site_xpos[tip].copy()

    probes = _flexion_joints(model, side, finger)
    if not probes:
        raise RuntimeError(f"no flexion joints found for {side} {finger}")
    for adr, val in probes:
        data.qpos[adr] = val
    mujoco.mj_kinematics(model, data)
    d = data.site_xpos[tip] - open_pos
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        raise RuntimeError(f"flexion probe moved {side} {finger} by {n:.2e} m")
    return d / n


def _surface_world(model, data, body_names: list[str]) -> np.ndarray:
    """Visual-mesh vertices of the given bodies, in world coords at ``data``."""
    from retargeting.utils.in_hand import extract_body_mesh_verts
    parts = [extract_body_mesh_verts(model, _bid(model, nm), data=data,
                                     apply_geom_xform=True) for nm in body_names]
    return np.concatenate(parts, axis=0)


def body_mesh_world(model, data, body_id: int):
    """Visual-mesh vertices *and* triangles of one body, in world coords."""
    import mujoco
    from retargeting.utils.mujoco_utils import quat_wxyz_to_rotmat
    V, F, base = [], [], 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body_id:
            continue
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        v = np.asarray(model.mesh_vert[va:va + vn], float)
        f = np.asarray(model.mesh_face[fa:fa + fn], np.int64)
        v = v @ quat_wxyz_to_rotmat(model.geom_quat[g]).T + model.geom_pos[g]
        v = v @ data.xmat[body_id].reshape(3, 3).T + data.xpos[body_id]
        V.append(v)
        F.append(f + base)
        base += vn
    if not V:
        return None, None
    return np.concatenate(V), np.concatenate(F)


def _surface_mesh_world(model, data, body_names: list[str]):
    """Merged triangle mesh of the given bodies, in world coords at ``data``."""
    V, F, base = [], [], 0
    for nm in body_names:
        v, f = body_mesh_world(model, data, _bid(model, nm))
        if v is None:
            continue
        V.append(v)
        F.append(f + base)
        base += len(v)
    return np.concatenate(V), np.concatenate(F)


def _distal_world(model, data, spec) -> np.ndarray:
    if isinstance(spec, tuple):
        return data.site_xpos[_sid(model, spec[1])].copy()
    return data.xpos[_bid(model, spec)].copy()


def segment_frame(model, data, origin_body: str, distal, palmar: np.ndarray,
                  surf: np.ndarray) -> dict:
    """Orthonormal frame plus extents for one segment, in world coords.

    ``x`` is the bone axis, ``z`` the palmar direction orthogonalised against
    it, ``y`` completes the triad. The y and z centres are recorded so a
    segment whose skin sits off the bone axis maps centre to centre rather than
    axis to axis.

    The x scale is how far the *skin* reaches along the bone, not the distance
    to the distal joint. Those two agree on MANO but not on this robot, whose
    ``*_tip`` sites sit mid-phalanx: the index distal link reaches 1.9 bone
    lengths while its tip site is at 1.0. Normalising by the joint distance
    therefore landed MANO's fingertip halfway down the robot's finger and left
    the distal 45% of the pad — the part that actually touches — unreachable.
    Anchoring x at the proximal joint and scaling by the skin's reach makes
    both ends shared landmarks; the two skins are nearly the same length
    (2.57 cm robot vs 2.74 cm MANO on the index) even though the bones are not.
    """
    o = data.xpos[_bid(model, origin_body)].copy()
    bone = _distal_world(model, data, distal) - o
    L = float(np.linalg.norm(bone))
    if L < 1e-6:
        raise RuntimeError(f"degenerate bone axis on {origin_body}")
    x = bone / L

    z = palmar - x * float(palmar @ x)
    nz = float(np.linalg.norm(z))
    if nz < 1e-6:
        raise RuntimeError(f"palmar direction parallel to bone on {origin_body}")
    z = z / nz
    y = np.cross(z, x)

    rel = surf - o
    a, b = rel @ y, rel @ z
    cy, cz_ = float(np.median(a)), float(np.median(b))

    def half(v, c):
        """Reach on each side of the centre, measured separately.

        A single symmetric half-width assumes the two sides are alike. They are
        not: MANO's palm is a rounded blob whose palmar skin sits around 0.2 of
        its half-width, while this robot's palm is a flat plate whose palmar
        face sits near 0.9 of its own. Sharing one scale sent MANO's palm
        points to a depth *inside* the robot's palm, where the snap — with no
        well-defined radial direction that close to the axis — scattered them.
        Normalising each side by its own reach maps skin to skin.
        """
        pos = v[v > c] - c
        neg = c - v[v < c]
        p = float(np.percentile(pos, _EXTENT_PCT[1])) if len(pos) else 0.0
        n = float(np.percentile(neg, _EXTENT_PCT[1])) if len(neg) else 0.0
        return max(p, 1e-4), max(n, 1e-4)

    sy_p, sy_n = half(a, cy)
    sz_p, sz_n = half(b, cz_)
    reach = float(np.percentile(rel @ x, _EXTENT_PCT[1]))
    return dict(
        origin=o, x=x, y=y, z=z,
        bone=L,
        sx=max(reach, 1e-4),
        cy=cy, sy_p=sy_p, sy_n=sy_n,
        cz=cz_, sz_p=sz_p, sz_n=sz_n,
        # Kept for reporting only; the transfer uses the signed halves.
        sy=(sy_p + sy_n) / 2.0, sz=(sz_p + sz_n) / 2.0,
        R=np.stack([x, y, z], axis=0),      # world -> frame
    )


def _signed(v: np.ndarray, sp: float, sn: float) -> np.ndarray:
    return np.where(v >= 0.0, v / sp, v / sn)


def _unsigned(u: np.ndarray, sp: float, sn: float) -> np.ndarray:
    return np.where(u >= 0.0, u * sp, u * sn)


def _normalise(pts: np.ndarray, fr: dict) -> np.ndarray:
    rel = pts - fr["origin"]
    return np.stack([
        (rel @ fr["x"]) / fr["sx"],
        _signed((rel @ fr["y"]) - fr["cy"], fr["sy_p"], fr["sy_n"]),
        _signed((rel @ fr["z"]) - fr["cz"], fr["sz_p"], fr["sz_n"]),
    ], axis=1)


def _denormalise(u: np.ndarray, fr: dict) -> np.ndarray:
    return (fr["origin"]
            + np.outer(u[:, 0] * fr["sx"], fr["x"])
            + np.outer(_unsigned(u[:, 1], fr["sy_p"], fr["sy_n"]) + fr["cy"], fr["y"])
            + np.outer(_unsigned(u[:, 2], fr["sz_p"], fr["sz_n"]) + fr["cz"], fr["z"]))


def _radial(pts: np.ndarray, fr: dict) -> np.ndarray:
    """Unit direction from the bone axis out to each point."""
    rel = pts - fr["origin"]
    perp = rel - np.outer(rel @ fr["x"], fr["x"])
    return perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-12)


def snap_to_surface(pts: np.ndarray, surf: np.ndarray, fr: dict,
                    faces: np.ndarray | None = None) -> np.ndarray:
    """Move each point onto the robot's skin without moving it *along* the skin.

    The snap travels along the **palmar normal**, so it changes only how deep
    the point sits and leaves its position along the bone and across the
    segment alone — and those two coordinates are the whole content of the
    correspondence.

    Two alternatives were measured and are worse. Snapping to the nearest
    vertex drags the point sideways along the skin: 1.1 mm on a finger, 4.3 mm
    on the palm, against 0.4 and 2.4 mm for this cast. Snapping radially, out
    from the bone axis, is no better in-plane and loses side agreement on the
    distal phalanges besides. Radial is the natural direction only for a
    surface of revolution about the bone, where the surface normal *is* the
    radial direction; a finger is close to one (normal and radial differ by a
    median 25 deg) and a palm is not (42 deg), but neither is close enough for
    it to beat simply going straight out the palm side.

    Nearest-neighbour snapping does not have that property, and on the palm it
    is destructive. Human and robot palms are different shapes — a rounded blob
    against a flat plate with raised knuckle housings — so three scale numbers
    cannot reconcile them and the transferred points land a good 7 mm off the
    skin. Pulled to their nearest vertex, neighbouring points get dragged onto
    whatever local feature is closest and pile up there: measured on this hand,
    the transfer preserved MANO's even 7.6 +- 2.2 mm spacing and the
    nearest-neighbour snap then collapsed it to 5.6 +- 3.6 mm, clumps beside
    bare patches. The slab cast restores it exactly (7.6 +- 2.3 mm, and 105/96
    palmar/dorsal against MANO's own 105/96). Fingers barely noticed the
    difference either way (1.3 mm moves) because a phalanx and a finger link
    really are the same shape.

    ``faces`` enables the ray cast; without it this degrades to the old
    side-restricted nearest vertex, which is still correct for the fingers.
    """
    from scipy.spatial import cKDTree
    rad_p = _radial(pts, fr)
    out = np.array(pts, dtype=float, copy=True)

    hit = np.zeros(len(pts), bool)
    if faces is not None:
        import trimesh
        mesh = trimesh.Trimesh(surf, faces, process=False)
        # Start on the segment's mid-plane and walk outwards, keeping the last
        # surface met. Casting the other way — from outside inwards — sounds
        # equivalent but is worse: near the mid-plane a point's side is barely
        # determined, and entering from the wrong face drops palm side
        # agreement from 98% to 91%.
        depth = (pts - fr["origin"]) @ fr["z"] - fr["cz"]
        direction = np.outer(np.where(depth >= 0, 1.0, -1.0), fr["z"])
        start = pts - np.outer(depth, fr["z"])
        loc, ray_idx, _ = mesh.ray.intersects_location(
            start, direction, multiple_hits=True)
        if len(ray_idx):
            order = np.lexsort((np.linalg.norm(loc - start[ray_idx], axis=1), ray_idx))
            loc, ray_idx = loc[order], ray_idx[order]
            last = np.ones(len(ray_idx), bool)
            last[:-1] = ray_idx[:-1] != ray_idx[1:]
            out[ray_idx[last]] = loc[last]
            hit[ray_idx[last]] = True

    # Anything the ray missed — a point off the end of the link, or a gap in
    # the mesh — falls back to the nearest vertex on its own side of the bone.
    if not hit.all():
        rad_s = _radial(surf, fr)
        cos_t = np.cos(np.radians(_SNAP_CONE_DEG))
        tree_all = cKDTree(surf)
        for i in np.flatnonzero(~hit):
            keep = np.flatnonzero(rad_s @ rad_p[i] > cos_t)
            if len(keep) == 0:
                out[i] = surf[tree_all.query(pts[i])[1]]
                continue
            out[i] = surf[keep[int(np.argmin(
                np.linalg.norm(surf[keep] - pts[i], axis=1)))]]
    return out


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


def eval_points(body_ids, local_offset, xpos, xmat):
    """Map the table onto live poses: ``xpos[b] + xmat[b] @ offset``.

    ``xpos``/``xmat`` come straight from ``wp.to_torch(env.data_wp.xpos)`` and
    ``...xmat`` and keep their leading world axis, so this evaluates every
    sampled world at once. Torch and numpy inputs both work.
    """
    try:
        import torch
        is_torch = torch.is_tensor(xpos)
    except ImportError:
        is_torch = False

    if is_torch:
        import torch
        R = xmat.reshape(xpos.shape[0], -1, 3, 3)[:, body_ids]     # (W, N, 3, 3)
        o = xpos[:, body_ids]                                      # (W, N, 3)
        return o + torch.einsum("wnij,nj->wni", R, local_offset)

    R = np.asarray(xmat).reshape(np.asarray(xpos).shape[0], -1, 3, 3)[:, body_ids]
    o = np.asarray(xpos)[:, body_ids]
    return o + np.einsum("wnij,nj->wni", R, np.asarray(local_offset))
