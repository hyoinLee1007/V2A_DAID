"""Retargeting pipeline: preprocess reconstruction output, then physics-optimize onto a robot hand.

Usage: python launch.py --task whisking --raw-dir ../reconstruction/whisking
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import loguru
import numpy as np
import tyro

from retargeting.config import Config, filter_config_fields, load_config_yaml
from retargeting.pipeline.decompose_mesh import main as decompose_mesh
from retargeting.pipeline.generate_scene import main as generate_scene
from retargeting.pipeline.optimize_physics import main as optimize_physics
from retargeting.pipeline.process_dataset import main as process_dataset
from retargeting.pipeline.resolve_pedestal import resolve_scene_pedestal
from retargeting.pipeline.solve_ik import main as solve_ik

CONFIG_DIR = Path(__file__).parent / "config"

# Scene flags for the do_as_i_do dataset: the object rests on an auto-placed
# pedestal (with a welded support) rather than directly on the floor.
OBJECT_FLOOR_COLLISION = False
HAND_FLOOR_COLLISION = False
USE_PEDESTAL = True
USE_SUPPORT = True


@dataclass
class PipelineConfig:
    """raw_dir is the reconstruction output dir; task is the video name (e.g. whisking)."""

    raw_dir: str = ""
    task: str = ""
    hand_type: str = "auto"
    data_id: int = 0
    # Drop the first N demo frames so frame 0 is a tracked pre-grasp pose.
    # warmup_analytical_init assumes frame 0 is a grasp-like pose whose palm
    # normal points at the object; a demo that opens with the hand resting on
    # the table within warmup_min_clearance of the object breaks that
    # assumption (the hand gets backed off in a bad direction and the grasp
    # collapses onto the object body — see cupmove).
    start_idx: int = 0
    dataset_name: str = "do_as_i_do"
    # Ray-depth correction for the raw hand track, applied in process_dataset
    # before the reference is built. See utils/ray_depth_correction.py. Empty
    # leaves the reconstruction as-is.
    ray_depth_path: str = ""
    robot_type: str = "sharpa"
    seed: int = 0
    wait_on_finish: bool = True
    max_sim_steps: int = 0
    force: bool = True
    smoothing: bool = True
    show_viewer: bool = True
    output_root_dir: str = "outputs"
    add_ur3_arm: bool = True


def load_mjwp_config(**overrides) -> Config:
    """Build Config from YAML defaults + dataset override + caller overrides."""
    cfg_dict = load_config_yaml(str(CONFIG_DIR / "default.yaml"))

    override_path = CONFIG_DIR / "override" / "do_as_i_do.yaml"
    if override_path.exists():
        cfg_dict.update(load_config_yaml(str(override_path)))

    cfg_dict.update(overrides)

    filtered = filter_config_fields(cfg_dict)
    if "pair_margin_range" in filtered:
        filtered["pair_margin_range"] = tuple(filtered["pair_margin_range"])
    if "xy_offset_range" in filtered:
        filtered["xy_offset_range"] = tuple(filtered["xy_offset_range"])
    filtered.pop("noise_scale", None)

    return Config(**filtered)


def run_pipeline(cfg: PipelineConfig) -> None:
    if not cfg.raw_dir:
        raise ValueError(
            "--raw-dir is required (the reconstruction pipeline's output "
            "directory, e.g. ../reconstruction/whisking)"
        )
    if not cfg.task:
        raise ValueError("--task is required (the video name, e.g. whisking)")

    # The dataset override YAML is otherwise read at Stage 4.5, long after the
    # reference has been built. The ray-depth correction has to be applied
    # while the raw track is being loaded, so pick that one setting up early —
    # a CLI --ray-depth-path still wins.
    if not cfg.ray_depth_path:
        override_path = CONFIG_DIR / "override" / f"{cfg.dataset_name}.yaml"
        if override_path.exists():
            cfg.ray_depth_path = load_config_yaml(str(override_path)).get(
                "ray_depth_path", "")

    # Stage 1: dataset processing
    pipeline_task = process_dataset(
        raw_dir=cfg.raw_dir,
        output_root_dir=cfg.output_root_dir,
        task=cfg.task,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        dataset_name=cfg.dataset_name,
        force=cfg.force,
        ray_depth_path=cfg.ray_depth_path,
    )
    if pipeline_task is None:
        loguru.logger.error(f"{cfg.dataset_name} processing failed (no task_name returned)")
        sys.exit(1)

    # Optional trim: slice the MANO keypoints NPZ so every downstream consumer
    # (solve_ik's reference AND optimize_physics' in-hand gate masks) sees the
    # same shifted timeline. Trimming only the IK via solve_ik(start_idx=...)
    # would desynchronize the gates, so the data itself is cut here.
    if cfg.start_idx > 0:
        from retargeting.utils.io import get_processed_data_dir, resolve_auto_embodiment

        emb = cfg.hand_type
        if emb == "auto":
            emb = resolve_auto_embodiment(cfg.dataset_name, cfg.output_root_dir, pipeline_task)
        mano_npz = os.path.join(
            get_processed_data_dir(cfg.output_root_dir, cfg.dataset_name, "mano", emb, pipeline_task, cfg.data_id),
            "trajectory_keypoints.npz",
        )
        data = dict(np.load(mano_npz))
        if "trim_start_idx" in data:
            loguru.logger.warning(
                "MANO data already trimmed (start_idx={}); skipping re-trim. "
                "Re-run with --force to regenerate from raw first.",
                int(data["trim_start_idx"]),
            )
        else:
            wl, wr = data["qpos_wrist_left"], data["qpos_wrist_right"]
            n_frames = wl.shape[0] if wl.size else wr.shape[0]
            for k, a in data.items():
                if a.ndim >= 1 and a.shape[0] == n_frames:
                    data[k] = a[cfg.start_idx:]
            data["trim_start_idx"] = np.array(cfg.start_idx)
            np.savez(mano_npz, **data)
            loguru.logger.info(
                "Trimmed first {} frames from {} ({} -> {} frames).",
                cfg.start_idx, mano_npz, n_frames, n_frames - cfg.start_idx,
            )

    # Stage 2: convex decomposition
    decompose_mesh(
        task=pipeline_task,
        dataset_name=cfg.dataset_name,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        thicken=0.002,
        dilate=0.002,
        force=cfg.force,
    )

    # Stage 3: XML generation
    generate_scene(
        task=pipeline_task,
        dataset_name=cfg.dataset_name,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        robot_type=cfg.robot_type,
        show_viewer=False,
        friction_scale=1.5,
        object_floor_collision=OBJECT_FLOOR_COLLISION,
        hand_floor_collision=HAND_FLOOR_COLLISION,
        use_pedestal=USE_PEDESTAL,
        use_support=USE_SUPPORT,
        force=cfg.force,
        add_ur3_arm=cfg.add_ur3_arm,
    )

    # Stage 4: inverse kinematics (runs against the pedestal-free scene_ik.xml)
    solve_ik(
        task=pipeline_task,
        dataset_name=cfg.dataset_name,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        robot_type=cfg.robot_type,
        show_viewer=False,
        force=cfg.force,
        smoothing=cfg.smoothing,
    )

    # Load the MJWP Config (YAML defaults + dataset override) before Stage 4.5:
    # it is the single source of truth for physics/threshold params, and the
    # pedestal step (Stage 4.5) needs hand_object_distance_thresh from it —
    # otherwise it falls back to in_hand.DEFAULT_DISTANCE_THRESH.
    config = load_mjwp_config(
        dataset_name=cfg.dataset_name,
        task=pipeline_task,
        data_id=cfg.data_id,
        robot_type=cfg.robot_type,
        embodiment_type=cfg.hand_type,
        seed=cfg.seed,
        wait_on_finish=cfg.wait_on_finish,
        max_sim_steps=cfg.max_sim_steps,
        force=cfg.force,
        show_viewer=cfg.show_viewer,
    )

    # Stage 4.5: resolve scene_ik.xml -> scene.xml (+ scene_eq.xml).
    resolve_scene_pedestal(
        output_root_dir=cfg.output_root_dir,
        dataset_name=cfg.dataset_name,
        robot_type=cfg.robot_type,
        embodiment_type=cfg.hand_type,
        task=pipeline_task,
        data_id=cfg.data_id,
        use_pedestal=USE_PEDESTAL,
        use_support=USE_SUPPORT,
        hand_object_distance_thresh=config.hand_object_distance_thresh,
        force=cfg.force,
        force_pedestal_start=config.warmup_min_clearance > 0.0,
    )

    # Stage 5: physics optimization (MuJoCo Warp)
    optimize_physics(config)


if __name__ == "__main__":
    cfg = tyro.cli(PipelineConfig)
    run_pipeline(cfg)
