#!/usr/bin/env python3
"""
verify_grasp_physics.py

Do as I Do의 trajectory_mjwp.npz를 단순 시각화하지 않고,
선택한 프레임에서 물체 pose 덮어쓰기를 중단한 뒤 실제 MuJoCo 물리를 진행하여
파지가 유지되는지 검증한다.

검증 모드
1) follow:
   선택 프레임 이후 저장된 로봇 ctrl 시퀀스를 그대로 open-loop 실행한다.
   물체 qpos는 절대 덮어쓰지 않는다.

2) hold:
   선택 프레임의 로봇 ctrl을 고정하고 정적 파지 안정성을 검사한다.

프레임 자동 선택
--frame -1을 사용하면 다음 조건에 가까운 프레임을 자동 탐색한다.
- warmup 이후
- 엄지-물체 접촉 없음
- 다른 손가락/손바닥-물체 접촉은 존재
- 물체가 초기 높이보다 충분히 떠 있음

사용 예시
python verify_grasp_physics.py \
  --run-dir outputs/sharpa/right/whisking/0 \
  --frame -1 \
  --mode follow \
  --seconds 2.0 \
  --viewer \
  --pause-before-start

필요 패키지
pip install mujoco numpy matplotlib pyyaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

LOGGER = logging.getLogger("verify_grasp_physics")


@dataclass
class Trajectory:
    qpos: np.ndarray
    qvel: np.ndarray | None
    ctrl: np.ndarray | None
    sim_step: np.ndarray | None


@dataclass
class ContactStats:
    total_object_hand: int
    thumb_object: int
    non_thumb_object: int
    normal_force_sum: float
    max_normal_force: float
    floor_object: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Do as I Do 저장 궤적의 실제 접촉 안정성을 MuJoCo로 재검증합니다."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/sharpa/right/whisking/0"),
        help="scene.xml과 trajectory_mjwp.npz가 있는 실행 디렉터리",
    )
    parser.add_argument("--scene", type=Path, default=None, help="scene.xml 직접 지정")
    parser.add_argument("--traj", type=Path, default=None, help="trajectory_mjwp.npz 직접 지정")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="-1이면 엄지 접촉이 사라진 후보 프레임을 자동 탐색",
    )
    parser.add_argument(
        "--mode",
        choices=("follow", "hold"),
        default="follow",
        help="follow=이후 저장 ctrl 실행, hold=현재 ctrl 고정",
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="물리 검증 시간")
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.05,
        help="초기 높이 대비 이 값 이상 하강하면 낙하로 판정[m]",
    )
    parser.add_argument(
        "--elevation-threshold",
        type=float,
        default=0.03,
        help="자동 프레임 탐색 시 물체가 기준 높이보다 떠 있어야 하는 최소 높이[m]",
    )
    parser.add_argument(
        "--object-body",
        type=str,
        default=None,
        help="물체 body 이름. 미지정 시 이름에 object가 포함된 free-joint body 자동 선택",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="미지정 시 run-dir/config.yaml에서 읽고, 실패하면 0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("physics_verification"),
        help="CSV, JSON, 그래프 저장 디렉터리",
    )
    parser.add_argument(
        "--scan-stride",
        type=int,
        default=1,
        help="자동 프레임 검색 간격. 긴 궤적이면 5~10 권장",
    )
    parser.add_argument(
        "--zero-object-velocity",
        action="store_true",
        help="선택 프레임에서 물체의 선속도·각속도를 0으로 설정",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="진행 로그 출력 step 간격",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="MuJoCo 3D viewer를 열어 실제 물리 rollout을 실시간으로 표시",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="viewer 재생 속도 배율. 1.0=실시간, 0.5=절반 속도, 2.0=2배속",
    )
    parser.add_argument(
        "--pause-before-start",
        action="store_true",
        help="viewer를 연 뒤 Enter 입력 전까지 시뮬레이션 시작을 대기",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def flatten_chunked(
    value: np.ndarray,
    sim_step: np.ndarray | None,
) -> np.ndarray:
    """(chunk, step, dim)을 시간순 (T, dim)으로 평탄화한다."""
    array = np.asarray(value)
    if array.ndim == 3:
        if sim_step is not None and len(sim_step) == array.shape[0]:
            order = np.argsort(np.asarray(sim_step).reshape(-1))
            array = array[order]
        array = array.reshape(-1, array.shape[-1])
    elif array.ndim == 2:
        pass
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    else:
        raise ValueError(f"지원하지 않는 trajectory 배열 차원: {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float64)


def load_trajectory(path: Path) -> Trajectory:
    if not path.exists():
        raise FileNotFoundError(f"trajectory 파일 없음: {path}")

    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        if "qpos" not in keys:
            raise KeyError(f"'qpos'가 없습니다. 저장 키: {sorted(keys)}")

        sim_step = np.asarray(data["sim_step"]).reshape(-1) if "sim_step" in keys else None
        qpos = flatten_chunked(data["qpos"], sim_step)
        qvel = flatten_chunked(data["qvel"], sim_step) if "qvel" in keys else None
        ctrl = flatten_chunked(data["ctrl"], sim_step) if "ctrl" in keys else None

    lengths = {"qpos": len(qpos)}
    if qvel is not None:
        lengths["qvel"] = len(qvel)
    if ctrl is not None:
        lengths["ctrl"] = len(ctrl)

    min_length = min(lengths.values())
    if len(set(lengths.values())) > 1:
        LOGGER.warning("trajectory 길이가 다릅니다: %s. 최소 길이 %d로 자릅니다.", lengths, min_length)

    return Trajectory(
        qpos=qpos[:min_length],
        qvel=None if qvel is None else qvel[:min_length],
        ctrl=None if ctrl is None else ctrl[:min_length],
        sim_step=sim_step,
    )


def load_warmup_steps(config_path: Path) -> int:
    if not config_path.exists():
        return 0

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return int(config.get("warmup_steps", 0) or 0)
    except Exception as exc:
        LOGGER.warning("config.yaml의 warmup_steps를 읽지 못했습니다: %s", exc)
        return 0


def name_of(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return name or ""


def descendants(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    result: set[int] = set()
    for body_id in range(model.nbody):
        current = body_id
        while current > 0:
            if current == root_body_id:
                result.add(body_id)
                break
            current = int(model.body_parentid[current])
    result.add(root_body_id)
    return result


def find_object_body(model: mujoco.MjModel, requested_name: str | None) -> int:
    if requested_name:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, requested_name)
        if body_id < 0:
            raise ValueError(f"물체 body를 찾지 못했습니다: {requested_name}")
        return int(body_id)

    candidates: list[tuple[int, str]] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        joint_name = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if "object" in body_name.lower() or "object" in joint_name.lower():
            candidates.append((body_id, body_name))

    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0][0]
    if not unique:
        raise RuntimeError(
            "이름에 object가 포함된 free-joint body를 찾지 못했습니다. "
            "--object-body로 직접 지정하십시오."
        )

    raise RuntimeError(
        "물체 후보가 여러 개입니다. --object-body로 지정하십시오: "
        + ", ".join(name for _, name in unique)
    )


def find_free_joint_for_body(model: mujoco.MjModel, body_id: int) -> int:
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) != body_id:
            continue
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return joint_id
    raise RuntimeError("선택한 물체 body에 free joint가 없습니다.")


def object_geom_ids(model: mujoco.MjModel, object_body_id: int) -> set[int]:
    body_ids = descendants(model, object_body_id)
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def classify_hand_geom(model: mujoco.MjModel, geom_id: int) -> tuple[bool, bool]:
    """
    반환값: (hand 여부, thumb 여부)
    visual geom보다 collision_hand_* 이름을 우선 사용한다.
    """
    geom_name = name_of(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
    body_id = int(model.geom_bodyid[geom_id])
    body_name = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body_id).lower()
    combined = f"{geom_name} {body_name}"

    hand_tokens = ("hand", "thumb", "index", "middle", "ring", "pinky", "palm")
    is_hand = any(token in combined for token in hand_tokens)
    is_thumb = "thumb" in combined
    return is_hand, is_thumb


def is_floor_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    geom_name = name_of(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
    body_name = name_of(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        int(model.geom_bodyid[geom_id]),
    ).lower()
    combined = f"{geom_name} {body_name}"
    return any(token in combined for token in ("floor", "ground", "table", "pedestal"))


def contact_stats(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geoms: set[int],
) -> ContactStats:
    total = 0
    thumb = 0
    non_thumb = 0
    force_sum = 0.0
    max_force = 0.0
    floor_contact = 0

    wrench = np.zeros(6, dtype=np.float64)

    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 in object_geoms:
            object_geom = geom1
            other_geom = geom2
        elif geom2 in object_geoms:
            object_geom = geom2
            other_geom = geom1
        else:
            continue

        if is_floor_geom(model, other_geom):
            floor_contact += 1

        is_hand, is_thumb = classify_hand_geom(model, other_geom)
        if not is_hand:
            continue

        total += 1
        if is_thumb:
            thumb += 1
        else:
            non_thumb += 1

        wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal_force = max(float(wrench[0]), 0.0)
        force_sum += normal_force
        max_force = max(max_force, normal_force)

    return ContactStats(
        total_object_hand=total,
        thumb_object=thumb,
        non_thumb_object=non_thumb,
        normal_force_sum=force_sum,
        max_normal_force=max_force,
        floor_object=floor_contact,
    )


def disable_object_welds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body_id: int,
) -> list[str]:
    disabled: list[str] = []

    if model.neq == 0:
        return disabled

    for eq_id in range(model.neq):
        obj1 = int(model.eq_obj1id[eq_id])
        obj2 = int(model.eq_obj2id[eq_id])
        eq_type = int(model.eq_type[eq_id])

        name = name_of(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
        references_object = obj1 == object_body_id or obj2 == object_body_id
        looks_like_object_weld = "object" in name.lower()

        if eq_type == int(mujoco.mjtEq.mjEQ_WELD) and (
            references_object or looks_like_object_weld
        ):
            if hasattr(data, "eq_active"):
                data.eq_active[eq_id] = 0
            if hasattr(model, "eq_active0"):
                model.eq_active0[eq_id] = 0
            disabled.append(name or f"equality_{eq_id}")

    return disabled


def object_position(data: mujoco.MjData, object_body_id: int) -> np.ndarray:
    return np.asarray(data.xpos[object_body_id], dtype=np.float64).copy()


def set_state_from_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: Trajectory,
    frame: int,
    free_joint_id: int,
    zero_object_velocity: bool,
) -> None:
    data.qpos[:] = trajectory.qpos[frame]

    if trajectory.qvel is not None:
        data.qvel[:] = trajectory.qvel[frame]
    else:
        data.qvel[:] = 0.0

    if trajectory.ctrl is not None and model.nu > 0:
        if trajectory.ctrl.shape[1] != model.nu:
            raise ValueError(
                f"ctrl 차원 {trajectory.ctrl.shape[1]} != model.nu {model.nu}"
            )
        data.ctrl[:] = trajectory.ctrl[frame]

    if zero_object_velocity:
        dof_adr = int(model.jnt_dofadr[free_joint_id])
        # free joint: 병진 3 + 회전 3
        data.qvel[dof_adr : dof_adr + 6] = 0.0

    data.time = 0.0
    mujoco.mj_forward(model, data)


def find_candidate_frame(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    object_body_id: int,
    object_geoms: set[int],
    warmup_steps: int,
    elevation_threshold: float,
    stride: int,
) -> tuple[int, dict]:
    data = mujoco.MjData(model)

    start = min(max(warmup_steps, 0), len(trajectory.qpos) - 1)

    # 기준 높이는 warmup 종료 부근의 최저값으로 잡는다.
    sample_end = min(len(trajectory.qpos), start + max(100, int(0.5 / model.opt.timestep)))
    heights: list[float] = []
    for frame in range(start, sample_end, max(1, stride)):
        data.qpos[:] = trajectory.qpos[frame]
        if trajectory.qvel is not None:
            data.qvel[:] = trajectory.qvel[frame]
        mujoco.mj_forward(model, data)
        heights.append(float(object_position(data, object_body_id)[2]))

    base_height = min(heights) if heights else 0.0
    best_fallback: tuple[int, dict] | None = None

    for frame in range(start, len(trajectory.qpos), max(1, stride)):
        data.qpos[:] = trajectory.qpos[frame]
        if trajectory.qvel is not None:
            data.qvel[:] = trajectory.qvel[frame]
        else:
            data.qvel[:] = 0.0
        if trajectory.ctrl is not None and model.nu > 0:
            data.ctrl[:] = trajectory.ctrl[frame]

        mujoco.mj_forward(model, data)

        position = object_position(data, object_body_id)
        stats = contact_stats(model, data, object_geoms)
        elevated = position[2] >= base_height + elevation_threshold

        metadata = {
            "frame": frame,
            "object_z": float(position[2]),
            "base_height": float(base_height),
            "elevated": bool(elevated),
            "object_hand_contacts": stats.total_object_hand,
            "thumb_contacts": stats.thumb_object,
            "non_thumb_contacts": stats.non_thumb_object,
            "normal_force_sum": stats.normal_force_sum,
        }

        # 사용자가 묘사한 상황과 가장 가까운 후보
        if elevated and stats.thumb_object == 0 and stats.non_thumb_object > 0:
            return frame, metadata

        # 엄지뿐 아니라 전체 손 접촉이 없는 더 심각한 후보도 기억
        if (
            best_fallback is None
            and elevated
            and stats.thumb_object == 0
            and stats.total_object_hand == 0
        ):
            best_fallback = (frame, metadata)

    if best_fallback is not None:
        LOGGER.warning(
            "다른 손가락 접촉이 남은 후보는 찾지 못해, 전체 접촉이 없는 후보를 선택합니다."
        )
        return best_fallback

    raise RuntimeError(
        "자동 조건에 맞는 프레임을 찾지 못했습니다. "
        "replay_viser.py에서 확인한 프레임 번호를 --frame으로 직접 지정하십시오."
    )


def simulate(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    start_frame: int,
    object_body_id: int,
    object_geoms: set[int],
    free_joint_id: int,
    mode: str,
    seconds: float,
    drop_threshold: float,
    zero_object_velocity: bool,
    log_every: int,
    viewer_enabled: bool,
    realtime_factor: float,
    pause_before_start: bool,
) -> tuple[list[dict], dict]:
    data = mujoco.MjData(model)

    set_state_from_frame(
        model=model,
        data=data,
        trajectory=trajectory,
        frame=start_frame,
        free_joint_id=free_joint_id,
        zero_object_velocity=zero_object_velocity,
    )
    disabled_welds = disable_object_welds(model, data, object_body_id)
    mujoco.mj_forward(model, data)

    if disabled_welds:
        LOGGER.info("비활성화한 object weld: %s", disabled_welds)
    else:
        LOGGER.info("활성 object weld를 찾지 못했거나 이미 비활성 상태입니다.")

    initial_position = object_position(data, object_body_id)
    initial_stats = contact_stats(model, data, object_geoms)

    dt = float(model.opt.timestep)
    steps = max(1, int(math.ceil(seconds / dt)))

    if realtime_factor <= 0:
        raise ValueError("--realtime-factor는 0보다 커야 합니다.")

    fixed_ctrl = data.ctrl.copy() if model.nu > 0 else None
    rows: list[dict] = []

    def record_row(step: int) -> None:
        position = object_position(data, object_body_id)
        stats = contact_stats(model, data, object_geoms)
        drop = float(initial_position[2] - position[2])

        rows.append(
            {
                "step": step,
                "time": float(data.time),
                "object_x": float(position[0]),
                "object_y": float(position[1]),
                "object_z": float(position[2]),
                "z_drop": drop,
                "object_hand_contacts": stats.total_object_hand,
                "thumb_contacts": stats.thumb_object,
                "non_thumb_contacts": stats.non_thumb_object,
                "normal_force_sum": stats.normal_force_sum,
                "max_normal_force": stats.max_normal_force,
                "floor_contacts": stats.floor_object,
            }
        )

    def apply_control(step: int) -> None:
        if model.nu == 0:
            return

        if mode == "follow":
            if trajectory.ctrl is None:
                raise RuntimeError(
                    "follow 모드에는 trajectory의 ctrl 배열이 필요합니다."
                )
            ctrl_frame = min(start_frame + step, len(trajectory.ctrl) - 1)
            data.ctrl[:] = trajectory.ctrl[ctrl_frame]
        else:
            data.ctrl[:] = fixed_ctrl

    def run_step(step: int) -> None:
        apply_control(step)

        # 핵심:
        # 저장된 qpos를 다시 대입하지 않는다.
        # 특히 물체 free-joint qpos는 MuJoCo 동역학에 의해 자유롭게 변한다.
        mujoco.mj_step(model, data)

        if log_every > 0 and step % log_every == 0:
            position = object_position(data, object_body_id)
            stats = contact_stats(model, data, object_geoms)
            drop = float(initial_position[2] - position[2])
            LOGGER.info(
                "step=%d time=%.3f z=%.4f drop=%.4f contacts=%d thumb=%d",
                step,
                data.time,
                position[2],
                drop,
                stats.total_object_hand,
                stats.thumb_object,
            )

    if viewer_enabled:
        try:
            from mujoco import viewer as mj_viewer
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo viewer를 불러오지 못했습니다. "
                "GUI 환경과 mujoco 설치 상태를 확인하십시오."
            ) from exc

        LOGGER.info(
            "MuJoCo viewer를 엽니다. 창을 닫으면 rollout도 종료됩니다."
        )

        with mj_viewer.launch_passive(model, data) as viewer:
            # 화면 갱신 전에 선택 프레임 초기 상태를 한 번 표시
            viewer.sync()

            if pause_before_start:
                print(
                    "\n선택 프레임의 초기 자세를 표시했습니다.\n"
                    "MuJoCo 창을 확인한 뒤 터미널에서 Enter를 누르면 물리 시뮬레이션을 시작합니다."
                )
                input()

            record_row(0)

            for step in range(steps):
                if not viewer.is_running():
                    LOGGER.warning("viewer 창이 닫혀 rollout을 중단합니다.")
                    break

                wall_start = time.perf_counter()

                run_step(step)
                record_row(step + 1)
                viewer.sync()

                # MuJoCo timestep 기준으로 실제 시간에 맞춰 재생
                target_wall_dt = dt / realtime_factor
                elapsed = time.perf_counter() - wall_start
                remaining = target_wall_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            # 마지막 상태를 잠깐 볼 수 있게 유지
            if viewer.is_running():
                LOGGER.info("rollout 종료. viewer 창을 닫으면 결과 저장을 계속합니다.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
    else:
        record_row(0)
        for step in range(steps):
            run_step(step)
            record_row(step + 1)

    final_position = object_position(data, object_body_id)
    max_drop = max(row["z_drop"] for row in rows)
    min_contacts = min(row["object_hand_contacts"] for row in rows)
    contact_loss_duration = sum(
        dt for row in rows if row["object_hand_contacts"] == 0
    )

    summary = {
        "start_frame": start_frame,
        "mode": mode,
        "viewer_enabled": viewer_enabled,
        "simulated_seconds": float(data.time),
        "requested_seconds": seconds,
        "simulation_timestep": dt,
        "realtime_factor": realtime_factor,
        "initial_object_position": initial_position.tolist(),
        "final_object_position": final_position.tolist(),
        "max_z_drop": float(max_drop),
        "drop_threshold": float(drop_threshold),
        "dropped": bool(max_drop >= drop_threshold),
        "initial_object_hand_contacts": initial_stats.total_object_hand,
        "initial_thumb_contacts": initial_stats.thumb_object,
        "initial_non_thumb_contacts": initial_stats.non_thumb_object,
        "minimum_object_hand_contacts": int(min_contacts),
        "contact_loss_duration": float(contact_loss_duration),
        "disabled_welds": disabled_welds,
        "gravity": np.asarray(model.opt.gravity, dtype=float).tolist(),
    }
    return rows, summary

def save_results(
    output_dir: Path,
    rows: list[dict],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "physics_rollout.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    try:
        import matplotlib.pyplot as plt

        times = [row["time"] for row in rows]
        z_values = [row["object_z"] for row in rows]

        plt.figure(figsize=(9, 4.5))
        plt.plot(times, z_values)
        plt.xlabel("Time [s]")
        plt.ylabel("Object Z [m]")
        plt.title("Free-physics object height")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "object_height.png", dpi=160)
        plt.close()

        contacts = [row["object_hand_contacts"] for row in rows]
        thumb = [row["thumb_contacts"] for row in rows]

        plt.figure(figsize=(9, 4.5))
        plt.plot(times, contacts, label="All hand-object contacts")
        plt.plot(times, thumb, label="Thumb-object contacts")
        plt.xlabel("Time [s]")
        plt.ylabel("Contact count")
        plt.title("Hand-object contacts during free physics")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "contact_count.png", dpi=160)
        plt.close()
    except Exception as exc:
        LOGGER.warning("그래프 저장을 건너뜁니다: %s", exc)

    LOGGER.info("결과 CSV: %s", csv_path)
    LOGGER.info("결과 JSON: %s", json_path)


def main() -> int:
    configure_logging()
    args = parse_args()

    run_dir = args.run_dir.resolve()
    scene_path = (args.scene or (run_dir / "scene.xml")).resolve()
    traj_path = (args.traj or (run_dir / "trajectory_mjwp.npz")).resolve()

    if not scene_path.exists():
        LOGGER.error("scene.xml을 찾지 못했습니다: %s", scene_path)
        return 2

    LOGGER.info("scene: %s", scene_path)
    LOGGER.info("trajectory: %s", traj_path)

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        trajectory = load_trajectory(traj_path)
    except Exception:
        LOGGER.exception("모델 또는 궤적 로드 실패")
        return 2

    if trajectory.qpos.shape[1] != model.nq:
        LOGGER.error(
            "qpos 차원 불일치: trajectory=%d, model.nq=%d",
            trajectory.qpos.shape[1],
            model.nq,
        )
        return 2

    try:
        object_body_id = find_object_body(model, args.object_body)
        object_name = name_of(model, mujoco.mjtObj.mjOBJ_BODY, object_body_id)
        free_joint_id = find_free_joint_for_body(model, object_body_id)
        object_geoms = object_geom_ids(model, object_body_id)
    except Exception:
        LOGGER.exception("물체 body/free joint 자동 탐색 실패")
        return 2

    LOGGER.info(
        "object body=%s(id=%d), object geoms=%d",
        object_name,
        object_body_id,
        len(object_geoms),
    )

    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else load_warmup_steps(run_dir / "config.yaml")
    )
    LOGGER.info("warmup_steps=%d", warmup_steps)

    frame = args.frame
    candidate_metadata: dict | None = None

    if frame < 0:
        try:
            frame, candidate_metadata = find_candidate_frame(
                model=model,
                trajectory=trajectory,
                object_body_id=object_body_id,
                object_geoms=object_geoms,
                warmup_steps=warmup_steps,
                elevation_threshold=args.elevation_threshold,
                stride=args.scan_stride,
            )
        except Exception:
            LOGGER.exception("자동 프레임 탐색 실패")
            return 2
        LOGGER.info("자동 선택 프레임=%d: %s", frame, candidate_metadata)

    if frame >= len(trajectory.qpos):
        LOGGER.error(
            "프레임 범위 초과: frame=%d, trajectory length=%d",
            frame,
            len(trajectory.qpos),
        )
        return 2

    output_dir = args.output_dir.resolve() / f"frame_{frame}_{args.mode}"

    try:
        rows, summary = simulate(
            model=model,
            trajectory=trajectory,
            start_frame=frame,
            object_body_id=object_body_id,
            object_geoms=object_geoms,
            free_joint_id=free_joint_id,
            mode=args.mode,
            seconds=args.seconds,
            drop_threshold=args.drop_threshold,
            zero_object_velocity=args.zero_object_velocity,
            log_every=args.log_every,
            viewer_enabled=args.viewer,
            realtime_factor=args.realtime_factor,
            pause_before_start=args.pause_before_start,
        )
    except Exception:
        LOGGER.exception("물리 rollout 실패")
        return 2

    if candidate_metadata is not None:
        summary["auto_candidate"] = candidate_metadata

    save_results(output_dir, rows, summary)

    print("\n" + "=" * 72)
    print("검증 결과")
    print("=" * 72)
    print(f"선택 프레임             : {summary['start_frame']}")
    print(f"실행 모드               : {summary['mode']}")
    print(f"초기 엄지 접촉 수       : {summary['initial_thumb_contacts']}")
    print(f"초기 기타 손 접촉 수    : {summary['initial_non_thumb_contacts']}")
    print(f"최대 물체 하강          : {summary['max_z_drop']:.4f} m")
    print(f"전체 접촉 소실 시간     : {summary['contact_loss_duration']:.4f} s")
    print(f"낙하 판정               : {'예' if summary['dropped'] else '아니오'}")
    print(f"중력                    : {summary['gravity']}")
    print(f"비활성화한 weld         : {summary['disabled_welds']}")
    print(f"결과 디렉터리           : {output_dir}")
    print("=" * 72)

    if summary["dropped"]:
        print(
            "판정: replay_viser 화면에서는 저장된 물체 pose가 재생되었지만, "
            "자유 물리에서는 해당 파지가 유지되지 않았습니다."
        )
    else:
        print(
            "판정: 물체 pose를 덮어쓰지 않아도 유지되었습니다. "
            "엄지 이외 접촉, caging/hooking 또는 높은 마찰이 지지했을 가능성이 큽니다."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
