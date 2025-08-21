 import torch
import numpy as np
import argparse
import pickle
import smplx

import sys
sys.path.append('/mnt/bn/souldance-codes/visualization')

from visualization.utils import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/bn/HumanML3D/body_models/")
    parser.add_argument("--model_type", type=str, default="smplx", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="NEUTRAL", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="318290_331317_slice22.pkl")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="318290_331317_slice22.bvh")
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE",
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    # names = [
    #     "Pelvis",
    #     "Left_hip",
    #     "Right_hip",
    #     "Spine1",
    #     "Left_knee",
    #     "Right_knee",
    #     "Spine2",
    #     "Left_ankle",
    #     "Right_ankle",
    #     "Spine3",
    #     "Left_foot",
    #     "Right_foot",
    #     "Neck",
    #     "Left_collar",
    #     "Right_collar",
    #     "Head",
    #     "Left_shoulder",
    #     "Right_shoulder",
    #     "Left_elbow",
    #     "Right_elbow",
    #     "Left_wrist",
    #     "Right_wrist",
    #     "Left_palm",
    #     "Right_palm",
    # ]

    names = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3"
]

    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    # import pdb;pdb.set_trace()
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:55,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 1
    
    scaling = None
    
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["q"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = 1.0  # (1,)
            trans = poses["pos"]  # (N, 3)
    
    else:
        raise Exception("This file type is not supported!")
    
    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    # positions[:,0] += trans * 10
    positions[:, 0] += trans
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations[:, :55],
        "positions": positions[:, :55],
        "offsets": offsets[:55],
        "parents": parents[:55],
        "names": names[:55],
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
    # if mirror:
    #     rots_mirror, trans_mirror = mirror_rot_trans(
    #             rots, trans, names, parents)
    #     positions_mirror = pos.copy()
    #     positions_mirror[:,0] += trans_mirror
    #     rotations_mirror = np.degrees(
    #         quat.to_euler(rots_mirror, order=order))
        
    #     bvh_data ={
    #         "rotations": rotations_mirror,
    #         "positions": positions_mirror,
    #         "offsets": offsets,
    #         "parents": parents,
    #         "names": names,
    #         "order": order,
    #         "frametime": 1 / fps,
    #     }
        
    #     output_mirror = output.split(".")[0] + "_mirror.bvh"
    #     bvh.save(output_mirror, bvh_data)


# def joints2bvh()

if __name__ == "__main__":
    args = parse_args()
    
    smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
             mirror = args.mirror, gender=args.gender,
             poses=args.poses, num_betas=args.num_betas, 
             fps=args.fps, output=args.output)
    
    print("finished!")