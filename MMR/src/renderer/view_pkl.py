
# https://blog.csdn.net/jacke121/article/details/137094973?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-137094973-blog-120309297.235^v43^pc_blog_bottom_relevance_base6&spm=1001.2101.3001.4242.1&utm_relevant_index=1
"""Demo code for running visualizer."""
import os
import pickle
 
import numpy as np
from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset
from aist_plusplus.visualizer import plot_on_video
from smplx import SMPL
import torch
 
FLAGS = flags.FLAGS
flags.DEFINE_string('smpl_dir', '/mnt/bn/HumanML3D/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl', 'input local dictionary that stores SMPL data.')
flags.DEFINE_string('video_name', 'gMH_sBM_cAll_d23_mMH0_ch08_slice9', 'input video name to be visualized.')
flags.DEFINE_string('save_dir', '/mnt/bn/MMR/renders', 'output local dictionary that stores AIST++ visualization.')
flags.DEFINE_enum('mode', 'SMPL', ['2D', '3D', 'SMPL', 'SMPLMesh'], 'visualize 3D or 2D keypoints, or SMPL joints on image plane.')
 
 
def main(_):
    # Parsing data info.
    # aist_dataset = AISTDataset(FLAGS.anno_dir)
    # video_path = os.path.join(FLAGS.video_dir, f'{FLAGS.video_name}.mp4')
    # seq_name, view = AISTDataset.get_seq_name(FLAGS.video_name)
    view_idx = 0#AISTDataset.VIEWS.index(view)
    file_path="/mnt/bn/code/EDGE/data/train/motions_sliced/gMH_sBM_cAll_d23_mMH0_ch08_slice9.pkl"
    if FLAGS.mode == 'SMPL':  # SMPL joints
 
        with open(file_path, "rb") as f:
            poses = pickle.load(f)
            if "smpl_poses" in poses:
                rots = poses["smpl_poses"]  # (N, 72)
                smpl_poses = rots.reshape(-1, 24 * 3)  # (N, 24, 3)
            elif "poses" in poses:
                rots = poses["poses"]
            elif "pos" in poses:
                rots = poses["q"]  # (N, 72)
                smpl_poses = rots.reshape(-1, 24 , 3)  # (N, 24, 3)
                smpl_trans = poses['pos']  # (N, 3)
            else:
                rots = poses["pred_thetas"]  # (N, 72)
                smpl_poses = rots.reshape(-1, 22*3)  # (N, 24, 3)
 
        smpl_scaling = poses.get('smpl_scaling',[1])
        smpl_scaling=np.asarray(smpl_scaling)
        # smpl_trans = poses['smpl_trans']  # (N, 3)
 
        # smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(motion_dir, seq_name)
        smpl = SMPL(model_path=FLAGS.smpl_dir, gender='MALE', batch_size=1)
        keypoints3d = smpl.forward(global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(), body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(), transl=torch.from_numpy(smpl_trans).float(), scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(), ).joints.detach().numpy()
 
        vals = keypoints3d*1
        out_path=os.path.join(FLAGS.save_dir,"gMH_sBM_cAll_d23_mMH0_ch08_slice9.npy")
        
        print('save frames', vals.shape[0], file_path)
        np.save(out_path, vals)
        
        # np.savez_compressed(out_path, joints_3d={"data": vals})
 
        nframes, njoints, _ = keypoints3d.shape
        # env_name = aist_dataset.mapping_seq2env[seq_name]
        # cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
        # keypoints2d = cgroup.project(keypoints3d)
        # keypoints2d = keypoints2d.reshape(9, nframes, njoints, 2)[view_idx]
 
    elif FLAGS.mode == 'SMPLMesh':  # SMPL Mesh
        import trimesh  # install by `pip install trimesh`
        import vedo  # install by `pip install vedo`
        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)
        smpl = SMPL(model_path=FLAGS.smpl_dir, gender='MALE', batch_size=1)
        vertices = smpl.forward(global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(), body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(), transl=torch.from_numpy(smpl_trans).float(), scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(), ).vertices.detach().numpy()[0]  # first frame
        faces = smpl.faces
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
 
        keypoints3d = AISTDataset.load_keypoint3d(aist_dataset.keypoint3d_dir, seq_name, use_optim=True)
        pts = vedo.Points(keypoints3d[0], r=20)  # first frame
 
        vedo.show(mesh, pts, interactive=True)
        exit()
 
app.run(main)