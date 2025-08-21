import os
import pickle
import pytorch3d.transforms as transforms
import numpy as np
import torch
import os
import pickle
import re
from visualization.utils.config import smplx_pose, smplx_bvh,smplx_bvh_souldance,smpl_pose,souldance_bvh_smpl


def get_euler_from_bvh_smplx(bvh_path):
    bvh_data = read_bvh(bvh_path)
    start_frame = bvh_data.index("MOTION\n") + 3
    total_frame = int(bvh_data[start_frame - 2].strip().split(':')[-1])
    # end_frame = get_end_frame(bvh_data, total_frame) + start_frame
    end_frame = total_frame + start_frame
    smplx_pkl = []
    for data_idx in range(start_frame, end_frame):
        data = list(map(float, re.split(r'\s+', bvh_data[data_idx].strip())))
        pose = data[3:]

        smpl_pose_data = data[:3]
        for pose_name in smplx_pose:
            idx = smplx_bvh_souldance.index(pose_name)
            smpl_pose_data += pose[idx*3:idx*3+3]

        smplx_pkl.append(smpl_pose_data)

    return smplx_pkl


def get_euler_from_bvh(bvh_path):
    bvh_data = read_bvh(bvh_path)
    start_frame = bvh_data.index("MOTION\n") + 3
    total_frame = int(bvh_data[start_frame - 2].strip().split(':')[-1])
    # end_frame = get_end_frame(bvh_data, total_frame) + start_frame
    end_frame = total_frame + start_frame
    smplx_pkl = []
    for data_idx in range(start_frame, end_frame):
        data = list(map(float, re.split(r'\s+', bvh_data[data_idx].strip())))
        pose = data[3:]

        smpl_pose_data = data[:3]
        for pose_name in smpl_pose:
            idx = souldance_bvh_smpl.index(pose_name)
            smpl_pose_data += pose[idx*3:idx*3+3]

        smplx_pkl.append(smpl_pose_data)

    return smplx_pkl


def read_bvh(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def convert_to_npz(body_bvh_path,face_data=None,has_face=True,nb_joints=52):
    save_npz_path = body_bvh_path.replace('.bvh', '.npz')
    smplx_pose = get_euler_from_bvh_smplx(body_bvh_path) if nb_joints==52 else get_euler_from_bvh(body_bvh_path)
    smplx_pose_arr = np.array(smplx_pose)
    root_trans = smplx_pose_arr[:, :3]
    euler_angles = torch.tensor(smplx_pose_arr[:, 3:], dtype=torch.float32)
    euler_angles = euler_angles.reshape(-1, nb_joints, 3) * (torch.pi / 180.0)

    # Convert euler angles to rotation matrix
    rot_mats = transforms.euler_angles_to_matrix(euler_angles, convention="ZYX")  # "ZXY"
    # Convert rotation matrix to axis-angle
    axis_angles = transforms.matrix_to_axis_angle(rot_mats).reshape(-1, nb_joints * 3).numpy()

    pose_zero = np.zeros((axis_angles.shape[0], 3*3))
    pose = np.concatenate((axis_angles[:,:22*3], pose_zero,axis_angles[:,22*3:]), axis=-1)
    if has_face:
        exp = face_data
        print('use flame!')
    else:
        exp = np.zeros((pose.shape[0], 100))
    # print(npz_file)
    np.savez(save_npz_path, betas=[0] * 300, poses=pose, expressions=exp, trans=root_trans,model='smplx2020', gender='neutral', mocap_frame_rate=30)


# convert_to_axis_angles(r"C:\Users\Admin\Downloads\gHO_sBM_cAll_d19_mHO0_ch03_slice4.npz")
# convert_bvh_to_npz_souldance(r"C:\Users\Admin\Downloads\samplx_test.bvh",nb_joints=52)