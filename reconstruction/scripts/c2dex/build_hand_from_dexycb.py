"""Build a do-as-i-do-format all_hand_meshes.npz directly from DexYCB ground truth.

DexYCB's pose.npz stores, per frame, a 51-D MANO pose vector per hand:
    pose_m[:, 0, :]  = [global_orient(3), hand_pose(45), trans(3)]  (axis-angle, meters)
already expressed in the master camera's own optical frame (same convention as
pose_y -- see build_layout_from_dexycb.py), which is exactly the "camera space"
retargeting/pipeline/process_dataset.py expects for {side}_joints/{side}_vertices.

Frames where the hand wasn't observed/annotated carry an all-zero pose_m row;
those are marked invalid (process_dataset.py's spike-cleaner interpolates
through them).

Runs MANO forward kinematics via the exact wrapper (hawor.utils.process.run_mano
/ run_mano_left) and joint convention (21-keypoint OpenPose ordering, 1552-face
closed-wrist topology) that reconstruction's own HaWoR step uses -- so the
output matches process_dataset.py's expectations bit-for-bit in convention,
just sourced from GT MANO params instead of HaWoR's video regressor.

Must run inside the `hawor` conda env, with HAWOR_DIR as the working directory
(the MANO config paths inside hawor.utils.process are relative).
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--meta-yml", required=True)
    ap.add_argument("--mano-yml", required=True, help="subject's calibrated MANO betas")
    ap.add_argument("--output", required=True, help="destination all_hand_meshes.npz")
    ap.add_argument("--hawor-dir", required=True, help="reconstruction/modules/HaWoR")
    args = ap.parse_args()

    os.chdir(args.hawor_dir)
    sys.path.insert(0, args.hawor_dir)
    from hawor.utils.process import run_mano, run_mano_left  # noqa: E402

    meta = yaml.safe_load(open(args.meta_yml))
    sides = meta["mano_sides"]
    assert len(sides) == 1, f"Expected a single-hand DexYCB sequence, got {sides}"
    side = sides[0]

    mano_calib = yaml.safe_load(open(args.mano_yml))
    betas = np.asarray(mano_calib["betas"], dtype=np.float64)  # (10,)

    d = np.load(args.pose_npz)
    pose_m = d["pose_m"][:, 0, :].astype(np.float64)  # (T, 51)
    T = pose_m.shape[0]

    valid = np.abs(pose_m).sum(axis=1) > 0  # zero row == unannotated frame
    print(f"{side} hand: {valid.sum()}/{T} valid frames")

    global_orient = pose_m[:, 0:3]
    hand_pose = pose_m[:, 3:48]
    trans = pose_m[:, 48:51]
    betas_bt = np.tile(betas, (T, 1))

    use_cuda = torch.cuda.is_available()
    device_trans = torch.from_numpy(trans).unsqueeze(0).float()          # (1,T,3)
    device_rot = torch.from_numpy(global_orient).unsqueeze(0).float()    # (1,T,3)
    device_pose = torch.from_numpy(hand_pose).unsqueeze(0).float()       # (1,T,45)
    device_betas = torch.from_numpy(betas_bt).unsqueeze(0).float()       # (1,T,10)

    run_fn = run_mano if side == "right" else run_mano_left
    out = run_fn(device_trans, device_rot, device_pose, is_right=None, betas=device_betas, use_cuda=use_cuda)

    joints = out["joints"][0].detach().cpu().numpy().astype(np.float64)      # (T,21,3)
    vertices = out["vertices"][0].detach().cpu().numpy().astype(np.float64)  # (T,778,3)

    # Same closed-wrist face patch used by demo.py / run_mano*, appended to the
    # standard MANO faces; mirrored winding for the left hand.
    from lib.models.mano_wrapper import MANO  # noqa: E402
    mano_cfg = dict(data_dir="_DATA/data/", model_path="_DATA/data/mano", gender="neutral",
                     num_hand_joints=15, create_body_pose=False)
    base_faces = MANO(**mano_cfg).faces
    faces_new = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79],
    ])
    faces_right = np.concatenate([base_faces, faces_new], axis=0)
    faces = faces_right if side == "right" else faces_right[:, [0, 2, 1]]
    faces = faces.astype(np.int32)

    out_data = {
        f"{side}_vertices": vertices,
        f"{side}_joints": joints,
        f"{side}_valid": valid,
        f"{side}_rot": global_orient,
        f"{side}_hand_pose": hand_pose,
        f"{side}_betas": betas_bt,
        f"{side}_faces": faces,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, **out_data)
    print(f"Saved {side} hand mesh data ({T} frames, {vertices.shape[1]} verts, "
          f"{faces.shape[0]} faces) -> {args.output}")


if __name__ == "__main__":
    main()
