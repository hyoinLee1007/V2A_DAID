"""Apply the ray-depth correction to the hand track before anything uses it.

A monocular reconstruction pins the hand well in the image and badly along the
camera ray, and on cupmove that error is a ~6 cm displacement almost entirely
along the ray: correcting it with one scalar per frame closes the fingertip gap
from 2.38 cm to 0.42 cm. This is CHOIR's Stage 2 in the part that matters here.

The correction has been used so far only to *extract* contact maps. The
reference the optimizer tracks was still built from the raw track, so the
reference put the hand where the reconstruction wrongly said it was, and the
contact terms then had to argue against it. Measured on that reference: 36% of
its hand-object contact is on the back of the fingers. Correcting the track
here instead means the reference is right to begin with, and the contact terms
only have to refine it.

A depth correction along the camera ray is a pure translation, so it moves
positions and leaves ``*_rot``, ``*_hand_pose`` and ``*_betas`` untouched —
the hand's articulation and shape are unchanged, only where it sits.

Frames the oracle could not solve (approach, release, tracking dropouts) are
filled by interpolating between the ones it could. Leaving them at zero would
step the hand by several centimetres between neighbouring frames, and the
pipeline's spike cleaner would then treat the correction itself as the fault.
"""

from __future__ import annotations

import numpy as np


def load_ray_depth(path: str, n_frames: int, ramp: int = 5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """``(z_star, solved)`` for ``n_frames``, from an oracle_ray_depth npz.

    Unsolved frames are held at **zero**, not interpolated across. Those frames
    are the approach and the release — the hand is legitimately away from the
    object there, and the oracle declined to correct them for that reason.
    Interpolating between the solved frames on either side reinstates exactly
    the correction that was refused, and pulls the hand onto the object during
    the reach: measured on cupmove, the approach went from a 4.63 cm gap to
    0.15 cm and gained 86 anchors a frame where the raw track had none. The
    demonstrated reach then no longer exists for the optimizer to follow, and
    it arrives at the object from whatever direction the contact terms prefer.

    ``ramp`` frames on each side of a solved run blend the correction in, so
    the hand is not teleported several centimetres between two frames. The
    pipeline's own spike cleaner would otherwise read that step as the fault.
    """
    d = np.load(path)
    z = np.asarray(d["z_star"], float)
    good = (np.asarray(d["good"], bool) if "good" in d.files
            else np.isfinite(z))
    good = good & np.isfinite(z)

    if len(z) < n_frames:
        z = np.pad(z, (0, n_frames - len(z)), constant_values=np.nan)
        good = np.pad(good, (0, n_frames - len(good)), constant_values=False)
    z, good = z[:n_frames], good[:n_frames]

    out = np.where(good, np.nan_to_num(z), 0.0)
    if ramp > 0 and good.any():
        # Triangular blend of the zero-filled signal. Deliberately *not*
        # normalised by the local count of solved frames: doing that hands an
        # unsolved frame next to a solved one almost the full correction, which
        # is the leak this whole change exists to close — approach frames 14-17
        # were picking up -14.7 cm from their neighbour at 18. Convolving the
        # zeros in instead makes the correction fade over `ramp` frames, which
        # is the taper that was wanted.
        k = np.concatenate([np.arange(1, ramp + 1), [ramp + 1],
                            np.arange(ramp, 0, -1)]).astype(float)
        out = np.convolve(out, k / k.sum(), mode="same")
    return out, good


def apply_ray_depth(verts: np.ndarray, joints: np.ndarray, z_star: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate each frame along its own camera ray by ``z_star``.

    The ray is the direction from the camera to the hand. The reconstruction
    lives in the camera's own frame, so the camera is the origin and the ray is
    just the normalised hand centroid — no calibration needed.

    Vertices and joints are moved by the *same* vector, taken from the vertex
    centroid, so the hand stays rigid: this is a correction to where the hand
    is, not to how it is posed.

    Returns the corrected arrays and the per-frame shift actually applied.
    """
    n = verts.shape[0]
    shift = np.zeros((n, 3))
    c = verts.mean(axis=1)                                   # (N, 3)
    norm = np.linalg.norm(c, axis=1, keepdims=True)
    ray = c / np.maximum(norm, 1e-9)
    ok = (norm[:, 0] > 1e-6) & np.isfinite(z_star)
    shift[ok] = ray[ok] * z_star[ok, None]
    return verts + shift[:, None, :], joints + shift[:, None, :], shift


def correct_hand_track(verts: np.ndarray, joints: np.ndarray, path: str,
                       label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Load and apply in one step, logging what it did. Never raises.

    A missing or unreadable file leaves the track alone with a warning: this is
    a correction, and the pipeline must still run without one.
    """
    import loguru

    try:
        z, good = load_ray_depth(path, verts.shape[0])
    except (OSError, KeyError) as exc:
        loguru.logger.warning(
            "Ray-depth correction {}: could not read {} ({}); track left as-is.",
            label, path, exc)
        return verts, joints

    v, j, shift = apply_ray_depth(verts, joints, z)
    mag = np.linalg.norm(shift, axis=1)
    loguru.logger.info(
        "Ray-depth correction {}: {}/{} frames solved, shift mean {:.1f} mm "
        "max {:.1f} mm (z* mean {:+.1f} mm)",
        label, int(good.sum()), len(z), 1000 * mag.mean(), 1000 * mag.max(),
        1000 * float(np.mean(z[good])) if good.any() else 0.0)
    return v, j
