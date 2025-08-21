# Copyright (c) Bytedance

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright (c) [2025] [Bytedance]
# Copyright (c) [2025] [Xiaojie Li] 
# This file has been modified by Xiaojie Li on 2025/07/30

from omegaconf import DictConfig
import logging
import hydra
import src.prepare  # noqa
import pdb
import os
import torch
import json
import numpy as np
import pickle
from src.renderer.matplotlib import MatplotlibRender
from vis import skeleton_render
from src.config import read_config
from src.load import load_model_from_cfg
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from src.data.collate import collate_x_dict
from src.model.mmr import get_score_matrix
import sys
sys.path.append('/mnt/bn/MMR')

logger = logging.getLogger(__name__)

pick_music_as_you_like ="612810_623961_slice231.npy"

music_feats_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/jukebox_feats/'


@hydra.main(version_base=None, config_path="configs", config_name="text_motion_sim")
def music_motion_sim(cfg:DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name

    cfg = read_config(run_dir)
    seed_everything(cfg.seed)
    logger.info("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)
    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    with torch.inference_mode():
        # motion -> latent
        with open('/mnt/bn/MMR/outputs/mmr_1009_1/latents/mmrdata_all.npy', 'rb') as f:
            lat_m_all = torch.from_numpy(np.load(f)).to(dtype=torch.float, device=device)
        
        # import pdb
        # pdb.set_trace()
        text_x_dict = collate_x_dict([text_model(os.path.join(music_feats_dir,pick_music_as_you_like))])  # 1*150*4800
        lat_t = model.encode(text_x_dict, sample_mean=True)[0] # 1*256
        score = get_score_matrix(lat_t, lat_m_all).cpu()

    # an
    score_list = score.tolist()
    score_sorted = sorted(score_list, reverse=True)
    pick_score = score_sorted[:8] + score_sorted[-4:]

    motion_name_idx = [score_list.index(item) for item in pick_score]
    print(motion_name_idx)
    logger.info(
        f"The similariy score s (0 <= s <= 1) between the music and the motion is: {pick_score}"
    )

    render_vis(motion_name_idx, pick_score)

    

def render_vis(motion_idx_list, pick_score):
    logger.info("Start rendering:")
    # import pdb;pdb.set_trace()
    audio_path = find_music_file(pick_music_as_you_like.replace('.npy', '.wav')) 

    print('audio_path:',audio_path)
    for idx in range(len(motion_idx_list)):
        with open('./outputs/mmr_1009_1/latents/mmrdata_index_keyids_all.json', 'r') as f:
            key_map = json.load(f)
            index_key_id = key_map.get(str(motion_idx_list[idx]))
        with open('./datasets/annotations/mmrdance/annotations.json', 'r') as f:
            motion_map = json.load(f)
            motion_name = motion_map.get(index_key_id).get('path')
        motion_name = motion_name.split('/')[-1]
        print('motion_name:',motion_name)
        motion_path_pkl = find_pkl_file(motion_name.replace('.npy', '.pkl'))
        print(motion_path_pkl)
        conert_output = '/mnt/bn/MMR/outputs/mmr_1009_1/converts'
        convert_pkl_to_smpl(smpl_dir='.//body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl',pkl_file_path=motion_path_pkl,motion_name=motion_name,save_dir=conert_output,mode='SMPL')

        motion_path_npy = os.path.join(conert_output, motion_name) 
        print('motion_path_npy:', motion_path_npy)
        render_output = '/mnt/bn/MMR/outputs/mmr_1009_1/renders'
        with open(motion_path_npy, 'rb') as f:
            motion_data = np.load(f)
            print('motion_data_shape:',motion_data.shape)

            # render_motion = MatplotlibRender()
            # render_motion(joints=motion_data,output=os.path.join(render_output, 'r_{}_{}.mp4'.format(pick_score[idx], motion_name.replace('.npy', ''))))

            # skeleton_render(poses=motion_data, out=render_output,audioname=audio_path, name='r_{}_{}.mp4'.format(pick_score[idx], motion_name.replace('.npy', '')))
            if 'finedance' in motion_path_pkl:
                FPS = 30
            else:
                FPS = 60
            print(FPS)
            # skeleton_render(poses=motion_data, out=render_output,audioname=audio_path, name='r_{}_{}.mp4'.format(pick_score[idx], str(pick_score[idx])), fps = )
            render_output = os.path.join(render_output, 'r_{}_{}.mp4'.format(motion_name, str(pick_score[idx])))
            sys.path.append('/mnt/bn/souldance-codes')
            from utils.plot_script import plot_3d_motion_music
            from utils.paramUtil import t2m_kinematic_chain
            plot_3d_motion_music(render_output,audio_path, t2m_kinematic_chain, motion_data, title=str(pick_score[idx]), fps=FPS)


    logger.info("Ending!")



def find_pkl_file(motion_name):
    prob_list = ['/mnt/bn/code/EDGE/data/train/motions_sliced', '/mnt/bn/code/EDGE/data/test/motions_sliced',
                 '/mnt/bn/code/EDGE/data/souldance/train/motions_sliced', '/mnt/bn/code/EDGE/data/souldance/test/motions_sliced',
                 '/mnt/bn/code/EDGE/data/finedance/train/motions_sliced', '/mnt/bn/code/EDGE/data/finedance/test/motions_sliced']
    for dir in prob_list:
        tmp_path = os.path.join(dir, motion_name)
        if os.path.exists(tmp_path):
            return tmp_path
    return None


def find_music_file(motion_name):
    prob_list = ['/mnt/bn/HumanTOMATO/datasets/souldance_data/music', '/mnt/bn/HumanTOMATO/datasets/finedance_data/music',
                 '/mnt/bn/HumanTOMATO/datasets/aistpp_data/music']
    for dir in prob_list:
        tmp_path = os.path.join(dir, motion_name)
        if os.path.exists(tmp_path):
            return tmp_path
    return None


def convert_pkl_to_smpl(smpl_dir,pkl_file_path,motion_name,save_dir,mode='SMPL'):
    import os
    import pickle
    import numpy as np
    from absl import app
    from absl import flags
    from aist_plusplus.loader import AISTDataset
    from aist_plusplus.visualizer import plot_on_video
    from smplx import SMPL
    import torch

    if mode == 'SMPL':  # SMPL joints
        with open(pkl_file_path, "rb") as f:
            poses = pickle.load(f)
            if "smpl_poses" in poses:
                rots = poses["smpl_poses"]  # (N, 72)
                smpl_poses = rots.reshape(-1, 24 * 3)  # (N, 24, 3)
                smpl_trans = poses['smpl_trans']
            elif "poses" in poses:
                rots = poses["poses"]
            elif "pos" in poses:
                # import pdb;pdb.set_trace()
                if len(poses["q"].shape) == 2:
                    rots = poses["q"][:,:22 * 3]
                else:
                    rots = poses["q"][:,:22,:]  # (N, 72)
                smpl_poses = rots.reshape(-1, 22 , 3)  # (N, 24, 3)
                smpl_trans = poses['pos']  # (N, 3)
            else:
                rots = poses["pred_thetas"]  # (N, 72)
                smpl_poses = rots.reshape(-1, 22*3)  # (N, 24, 3)

        # import pdb;pdb.set_trace()
        smpl_scaling = poses.get('smpl_scaling',[1])
        smpl_scaling=np.asarray(smpl_scaling)
        smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)
        # import pdb;pdb.set_trace()
        pose_zero = np.zeros((smpl_poses.shape[0], 2,3))
        smpl_poses = np.concatenate((smpl_poses, pose_zero), axis=1)

        keypoints3d = smpl.forward(global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(), body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(), transl=torch.from_numpy(smpl_trans).float(), scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(), ).joints.detach().numpy()
 
        vals = keypoints3d[:, :22, :]
        out_path=os.path.join(save_dir,motion_name)
        
        print('save frames', vals.shape[0], pkl_file_path)
        np.save(out_path, vals)


if __name__ == "__main__":
    music_motion_sim()
