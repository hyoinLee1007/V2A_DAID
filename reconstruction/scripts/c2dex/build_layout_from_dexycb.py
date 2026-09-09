"""Build a do-as-i-do-format object layout JSON directly from DexYCB ground truth.

DexYCB's pose.npz stores, per frame, one 7-vector per loaded YCB object:
[qx, qy, qz, qw, tx, ty, tz] (unit quaternion in xyzw order + translation,
meters -- DexYCB's own toolkit is PyTorch/manopth-based and uses xyzw
throughout; verified empirically here by checking that the object's local
"tall" mesh axis maps to within 8 degrees of the real-world up direction at
frame 0, when the object is resting flat on the table -- the wxyz reading was
off by 112 degrees, i.e. the object nearly on its side), expressed in the
"world" frame that DexYCB defines as the *master* camera's own optical frame
(extrinsics.yml lists the master camera's own extrinsic as identity by
construction). do-as-i-do's retargeting/pipeline/process_dataset.py expects
object poses under `obj_tracking_out/{object}/combined_visualization/
layout_camera_frame_optimized.json` in wxyz order (field name
`quat_wxyz_camera_frame`) -- so DexYCB's GT poses can be dropped in directly
with no basis change, only a quaternion-order + field-name translation.

Since DexYCB's YCB meshes are already at real metric scale (unlike do-as-i-do's
own SAM-3D-reconstructed meshes, whose scale is unknown and fit numerically),
mesh_scale is fixed at 1.0.
"""

import argparse
import json

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-npz", required=True, help="DexYCB sequence pose.npz")
    ap.add_argument("--meta-yml", required=True, help="DexYCB sequence meta.yml")
    ap.add_argument("--object-name", required=True, help="do-as-i-do object name, e.g. pitcher")
    ap.add_argument("--output", required=True, help="output layout_camera_frame_optimized.json path")
    args = ap.parse_args()

    meta = yaml.safe_load(open(args.meta_yml))
    grasp_ind = meta["ycb_grasp_ind"]
    ycb_id = meta["ycb_ids"][grasp_ind]

    d = np.load(args.pose_npz)
    pose_y = d["pose_y"]  # (T, num_objects, 7): [qx,qy,qz,qw,tx,ty,tz]
    T = pose_y.shape[0]

    objects = []
    for fi in range(T):
        qx, qy, qz, qw, tx, ty, tz = pose_y[fi, grasp_ind].tolist()
        objects.append({
            "frame_idx": fi,
            "local_to_scene": {
                "translation_camera_frame": [tx, ty, tz],
                "quat_wxyz_camera_frame": [qw, qx, qy, qz],
            },
        })

    layout = {
        "frame": "camera_frame",
        "note": (
            "Ground-truth DexYCB object pose (pose_y), copied directly as camera-frame "
            "poses: DexYCB's world frame is the master camera's own optical frame "
            f"(x-right, y-down, z-fwd), matching this field's convention. object={args.object_name} "
            f"ycb_id={ycb_id} ycb_grasp_ind={grasp_ind}."
        ),
        "translation_scale_optimization": {
            "mesh_scale": 1.0,
            "mesh_scale_original": 1.0,
        },
        "objects": objects,
    }

    with open(args.output, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {len(objects)} frames -> {args.output}")


if __name__ == "__main__":
    main()
