#!/usr/bin/env python3
"""replay_viser.py — replay a retargeted trajectory in a viser viewer.

Loads a generated MuJoCo scene (`scene.xml`) and an optimized trajectory
(`trajectory_mjwp.npz`) from a retargeting run directory and plays it back in the
browser — reusing the pipeline's own `retargeting.utils.viser_viewer` geometry /
posing code. This is a *view-only* replay of an existing result: it does NOT
re-run the optimization (unlike `launch.py`, whose viewer is tied to Stage 5).

Alongside the retargeted trajectory it overlays up to two reference layers,
when their files exist in/for the run directory:

1. **MANO reference** (orange) — the reconstruction-pipeline input: deforming
   MANO hand mesh + tracked object pose, from the Stage-1 output
   ``outputs/mano/<embodiment>/<task>/<id>/trajectory_keypoints.npz`` (already
   gravity-aligned and world-offset into the scene frame by process_dataset).
2. **IK reference** (transparent blue) — the kinematic solution
   ``trajectory_kinematic.npz``, drawn with viser_viewer's built-in ref-ghost
   scene copy.

Each layer has a GUI checkbox; Frame slider / Play / FPS drive all layers on a
shared timeline (the mjwp sim_dt timeline; references advance every
``ref_dt/sim_dt`` sim steps, holding frame 0 during warmup). Missing reference
files simply disable their layer — the shipped whisking demo (retargeted
trajectory only) plays back as before.

Run from the `retargeting/` directory in the `retargeting` conda env:

    conda activate retargeting

    # defaults to the whisking demo output on port 8081
    python replay_viser.py

    # or point at any run dir / explicit files / different port
    python replay_viser.py --run-dir outputs/sharpa/right/whisking/0 --port 8081
    python replay_viser.py --scene path/to/scene.xml --traj path/to/trajectory_mjwp.npz
    python replay_viser.py --no-skip-warmup        # also show the 600 warmup frames

Then open http://localhost:<port> and use the Frame slider / Play button.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from retargeting.utils import viser_viewer

MANO_COLOR = (255, 150, 40)  # orange: raw reconstruction reference
MANO_OPACITY = 0.55


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        default="outputs/sharpa/right/whisking/0",
        help="Run dir containing scene.xml + trajectory_mjwp.npz + config.yaml.",
    )
    p.add_argument("--scene", default=None, help="Override path to scene.xml.")
    p.add_argument("--traj", default=None, help="Override path to trajectory .npz.")
    p.add_argument("--port", type=int, default=8081, help="Viser server port.")
    p.add_argument("--fps", type=float, default=60.0, help="Initial playback FPS.")
    p.add_argument(
        "--play-reference",
        nargs="?",
        const="tracked",
        default=None,
        choices=("tracked", "raw"),
        help="Play a reference as the main trajectory instead of the optimizer's "
             "output. 'tracked' (the default when the flag is given bare) is "
             "qpos_ref, i.e. what the optimizer actually followed, after "
             "de-penetration and the warmup offset decay. 'raw' is qpos_ref_raw, "
             "the same reference before those corrections — pass both in turn to "
             "see what the corrections did. Neither is trajectory_kinematic.npz, "
             "the 78-frame raw IK output that the blue ghost layer draws.",
    )
    p.add_argument(
        "--skip-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the leading warmup frames (default: skip).",
    )
    return p.parse_args()


def _load_run_config(config_yaml: Path) -> dict:
    """Best-effort read of the run's saved config.yaml (warmup_steps, sim_dt, ref_dt)."""
    if not config_yaml.exists():
        return {}
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.load(str(config_yaml)))
    except Exception:
        return {}


def load_qpos(traj_path: Path, model_nq: int) -> np.ndarray:
    """Load + flatten the optimized qpos to (T, nq), ordered by sim step.

    `trajectory_mjwp.npz` stores qpos chunked as (n_chunks, steps_per_chunk, nq);
    chunks are sequential executed segments, so we order by `sim_step` and flatten.
    """
    d = np.load(str(traj_path), allow_pickle=True)
    if "qpos" not in d.files:
        raise SystemExit(f"'qpos' not in {traj_path} (keys: {list(d.files)})")
    qpos = np.asarray(d["qpos"])
    if qpos.ndim == 3:
        if "sim_step" in d.files and len(d["sim_step"]) == qpos.shape[0]:
            qpos = qpos[np.argsort(np.asarray(d["sim_step"]).ravel())]
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.shape[-1] != model_nq:
        raise SystemExit(
            f"qpos width {qpos.shape[-1]} != model.nq {model_nq} — "
            "scene.xml and trajectory .npz are from different runs?"
        )
    return np.ascontiguousarray(qpos, dtype=np.float64)


class ManoOverlay:
    """Orange MANO hand mesh(es) + tracked-object mesh from trajectory_keypoints.npz.

    Hand vertices deform per frame, so the hand mesh is re-uploaded on frame
    change (778 verts — cheap). The object mesh is rigid: uploaded once, posed
    via handle.position/wxyz.
    """

    def __init__(self, keypoints_path: Path, outputs_root: Path, task: str):
        self.hands: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # side -> (verts, faces)
        self.obj_qpos: np.ndarray | None = None
        self.obj_mesh: trimesh.Trimesh | None = None
        self.n_frames = 0
        self.visible = True
        self._hand_handles: dict[str, object] = {}
        self._hand_frame: dict[str, int] = {}
        self._obj_handle = None

        if not keypoints_path.exists():
            print(f"MANO reference not found ({keypoints_path}) — layer disabled.")
            return

        d = np.load(str(keypoints_path))
        for side in ("right", "left"):
            v, f = f"mano_verts_{side}", f"mano_faces_{side}"
            if v in d.files and d[v].size > 0:
                self.hands[side] = (
                    d[v].astype(np.float32),
                    d[f].astype(np.uint32),
                )
                self.n_frames = max(self.n_frames, d[v].shape[0])

        # Tracked object pose: sides carry identical copies; prefer a side we render.
        for side in (*self.hands.keys(), "right", "left"):
            key = f"qpos_obj_{side}"
            if key in d.files and d[key].size > 0:
                self.obj_qpos = d[key].astype(np.float64)
                self.n_frames = max(self.n_frames, self.obj_qpos.shape[0])
                break

        mesh_path = outputs_root / "assets" / "objects" / task / "visual.obj"
        if self.obj_qpos is not None and mesh_path.exists():
            try:
                m = trimesh.load_mesh(str(mesh_path))
                if not isinstance(m, trimesh.Trimesh):
                    m = m.dump(concatenate=True)
                self.obj_mesh = m
            except Exception as e:
                print(f"Could not load object mesh {mesh_path}: {e}")

    @property
    def available(self) -> bool:
        return bool(self.hands) or self.obj_qpos is not None

    def build(self, server) -> None:
        self._server = server
        if self.obj_mesh is not None:
            self._obj_handle = server.scene.add_mesh_simple(
                "/mano_ref/object",
                vertices=np.asarray(self.obj_mesh.vertices, dtype=np.float32),
                faces=np.asarray(self.obj_mesh.faces, dtype=np.uint32),
                color=MANO_COLOR,
                opacity=MANO_OPACITY,
            )

    def show_frame(self, frame_idx: int) -> None:
        if not self.available or not self.visible:
            return
        fi = max(0, min(self.n_frames - 1, int(frame_idx)))
        for side, (verts, faces) in self.hands.items():
            f = min(fi, verts.shape[0] - 1)
            if self._hand_frame.get(side) == f:
                continue
            # Deforming mesh: re-add at the same path replaces the old handle.
            self._hand_handles[side] = self._server.scene.add_mesh_simple(
                f"/mano_ref/hand_{side}",
                vertices=verts[f],
                faces=faces,
                color=MANO_COLOR,
                opacity=MANO_OPACITY,
            )
            self._hand_frame[side] = f
        if self._obj_handle is not None and self.obj_qpos is not None:
            q = self.obj_qpos[min(fi, self.obj_qpos.shape[0] - 1)]
            self._obj_handle.position = tuple(q[:3])
            self._obj_handle.wxyz = tuple(q[3:7])

    def set_visible(self, value: bool, frame_idx: int) -> None:
        self.visible = value
        for h in self._hand_handles.values():
            h.visible = value
        if self._obj_handle is not None:
            self._obj_handle.visible = value
        if value:
            # Hand meshes may be stale (frames skipped while hidden).
            self._hand_frame.clear()
            self.show_frame(frame_idx)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush progress even when piped to a file
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    scene_path = Path(args.scene).resolve() if args.scene else run_dir / "scene.xml"
    traj_path = Path(args.traj).resolve() if args.traj else run_dir / "trajectory_mjwp.npz"
    kin_path = run_dir / "trajectory_kinematic.npz"
    for pth in (scene_path, traj_path):
        if not pth.exists():
            raise SystemExit(f"Not found: {pth}")

    # outputs/<robot>/<embodiment>/<task>/<id> -> outputs root + task + embodiment.
    # Run dirs that don't follow this layout just get the MANO layer disabled.
    keypoints_path = None
    outputs_root = None
    task = ""
    try:
        data_id, task, embodiment = (
            run_dir.name, run_dir.parent.name, run_dir.parent.parent.name
        )
        outputs_root = run_dir.parents[3]
        keypoints_path = (
            outputs_root / "mano" / embodiment / task / data_id / "trajectory_keypoints.npz"
        )
    except IndexError:
        pass

    cfg = _load_run_config(run_dir / "config.yaml")
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    sim_dt = float(cfg.get("sim_dt", 0.005))
    ref_dt = float(cfg.get("ref_dt", 0.0333))
    ref_steps = max(1, int(round(ref_dt / sim_dt)))  # sim steps per reference frame

    # Build the MuJoCo model + data (spec is needed by build_and_log_scene_from_spec).
    spec = mujoco.MjSpec.from_file(str(scene_path))
    model = spec.compile()
    data = mujoco.MjData(model)

    if args.play_reference:
        key = "qpos_ref" if args.play_reference == "tracked" else "qpos_ref_raw"
        _d = np.load(str(traj_path), allow_pickle=True)
        if key not in _d.files:
            raise SystemExit(
                f"'{key}' not in {traj_path} — this run predates the change that "
                "saves the reference. Re-run the optimization.")
        qpos = np.ascontiguousarray(
            np.asarray(_d[key], np.float64).reshape(-1, model.nq))
        print(f"playing the {args.play_reference.upper()} REFERENCE "
              f"('{key}', {len(qpos)} steps), not the optimized trajectory")
    else:
        qpos = load_qpos(traj_path, model.nq)

    # IK reference (blue ghost) — optional.
    kin_qpos = None
    data_ref = None
    if kin_path.exists():
        kd = np.load(str(kin_path))
        kin_qpos = np.asarray(kd["qpos"], dtype=np.float64)
        if kin_qpos.ndim == 3:
            kin_qpos = kin_qpos.reshape(-1, kin_qpos.shape[-1])
        if kin_qpos.shape[-1] != model.nq:
            print(
                f"Kinematic qpos width {kin_qpos.shape[-1]} != model.nq {model.nq}"
                " — IK reference disabled."
            )
            kin_qpos = None
        else:
            data_ref = mujoco.MjData(model)

    start = warmup_steps if (args.skip_warmup and warmup_steps > 0) else 0
    start = min(start, max(0, len(qpos) - 1))
    n_frames = len(qpos) - start
    if n_frames <= 0:
        raise SystemExit("No frames to play.")

    def ref_frame_for(display_idx: int) -> int:
        """Map a display frame to the kinematic/MANO reference frame index."""
        sim_i = display_idx + start
        return max(0, sim_i - warmup_steps) // ref_steps

    # Spin up viser and upload the scene geometry (ref ghost only if IK data exists).
    viser_viewer.init_viser(app_name="retargeting-replay", port=args.port)
    server = viser_viewer._get_server()
    body_ids = viser_viewer.build_and_log_scene_from_spec(
        spec, model, xml_path=scene_path,
        build_ref=kin_qpos is not None, build_gui=False,
    )

    mano = (
        ManoOverlay(keypoints_path, outputs_root, task)
        if keypoints_path is not None
        else None
    )
    if mano is not None and mano.available:
        mano.build(server)
    else:
        mano = None

    def show_frame(frame_idx: int) -> None:
        fi = max(0, min(n_frames - 1, int(frame_idx)))
        data.qpos[:] = qpos[fi + start]
        mujoco.mj_kinematics(model, data)  # populate body xpos/xquat for log_frame
        rf = ref_frame_for(fi)
        if kin_qpos is not None:
            data_ref.qpos[:] = kin_qpos[min(rf, len(kin_qpos) - 1)]
            mujoco.mj_kinematics(model, data_ref)
        viser_viewer.log_frame(
            data, sim_time=0.0, viewer_body_entity_and_ids=body_ids,
            data_ref=data_ref, record=False,
        )
        if mano is not None:
            mano.show_frame(rf)

    # --- GUI: layer toggles + Frame slider + Play/Pause + FPS ---
    with server.gui.add_folder("Layers"):
        if mano is not None:
            cb_mano = server.gui.add_checkbox("MANO reference (orange)", initial_value=True)
        if kin_qpos is not None:
            cb_ik = server.gui.add_checkbox("IK reference (blue)", initial_value=True)
        cb_robot = server.gui.add_checkbox("Retargeted robot", initial_value=True)

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    play_button = server.gui.add_button("Play")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=120, step=1, initial_value=int(args.fps)
    )

    if mano is not None:

        @cb_mano.on_update
        def _(_) -> None:
            mano.set_visible(cb_mano.value, ref_frame_for(int(frame_slider.value)))

    if kin_qpos is not None:

        @cb_ik.on_update
        def _(_) -> None:
            # Ghost body frames parent their geoms, so toggling frames suffices.
            for h, _bid in viser_viewer._STATE.ref_body_handles:
                h.visible = cb_ik.value
            for gh in viser_viewer._STATE.ref_geom_handles:
                gh.visible = cb_ik.value

    @cb_robot.on_update
    def _(_) -> None:
        # Skip worldbody (bid 0) so the floor stays visible.
        for h, bid in body_ids:
            if bid != 0:
                h.visible = cb_robot.value

    playing = {"on": False}
    suppress_cb = {"on": False}  # don't double-render during programmatic advance

    def set_button_label(text: str) -> None:
        for attr in ("name", "label"):  # viser version compatibility
            try:
                setattr(play_button, attr, text)
                return
            except Exception:
                pass

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress_cb["on"]:
            show_frame(frame_slider.value)

    @play_button.on_click
    def _(_) -> None:
        playing["on"] = not playing["on"]
        set_button_label("Pause" if playing["on"] else "Play")

    def playback_loop() -> None:
        while True:
            if playing["on"]:
                nxt = int(frame_slider.value) + 1
                if nxt >= n_frames:
                    nxt = 0
                suppress_cb["on"] = True
                frame_slider.value = nxt
                suppress_cb["on"] = False
                show_frame(nxt)
                time.sleep(1.0 / max(1.0, float(fps_slider.value)))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback_loop, daemon=True).start()

    show_frame(0)
    print(f"Scene:       {scene_path}")
    print(f"Retargeted:  {traj_path}")
    print(f"IK ref:      {kin_path if kin_qpos is not None else 'not found'}")
    print(f"MANO ref:    {keypoints_path if mano is not None else 'not found'}")
    print(
        f"Playing {n_frames} frames"
        + (f" (skipped {start} warmup frames; --no-skip-warmup to include)" if start else "")
    )
    print(f"Viewer running at http://localhost:{args.port}  (Ctrl+C to exit)")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
