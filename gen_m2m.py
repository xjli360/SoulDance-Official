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

import os
from os.path import join as pjoin
import time

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalM2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical
from data.rotation_conversions import *


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion_music

from utils.paramUtil import t2m_kinematic_chain, t2m_body_hand_kinematic_chain
from visualization.utils.emage_npz import convert_to_npz

import numpy as np
import codecs as cs

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'latest.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='music',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=256,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='music',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=256,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'latest.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


if __name__ == '__main__':
    parser = EvalM2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    if opt.dataset_name == "aistpp":
        nb_joints = 22
        kinematic_chain = t2m_kinematic_chain
        FPS = 60
        audio_dir = '/mnt/bn/code/EDGE/data/all_music_sliced_5'
        val_split_file = '/mnt/bn/souldance-codes/dataset/aistpp/test45.txt'
        t_lens = 300
        dim_pose = 263
        num_part = 1
        model_type = 'smpl'
    elif opt.dataset_name == 'finedance':
        nb_joints = 22
        kinematic_chain = t2m_kinematic_chain
        FPS = 30
        audio_dir = '/mnt/bn/HumanTOMATO/datasets/finedance_data/music'
        val_split_file = '/mnt/bn/souldance-codes/dataset/finedance/tiny.txt'
        t_lens = 150
        dim_pose = 263
        model_type = 'smplx'
    elif opt.dataset_name == 'souldance':
        nb_joints = 52
        kinematic_chain = t2m_body_hand_kinematic_chain
        FPS = 60
        audio_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/music'
        val_split_file = '/mnt/bn/souldance-codes/dataset/souldance/test.txt'
        t_lens = 300
        dim_pose = 723
        num_part = 3
        model_type = 'smplx'

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    joints_all_dir = pjoin(result_dir, 'joints_all')
    motion_feats_dir = pjoin(result_dir, 'motion_feats')

    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)
    os.makedirs(joints_all_dir,exist_ok=True)
    os.makedirs(motion_feats_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)
    res_model.eval()
    res_model.to(opt.device)
    assert res_opt.vq_name == model_opt.vq_name
    m2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    m2m_transformer.eval()
    vq_model.eval()

    m2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    music_list = []
    name_list = []
    with cs.open(val_split_file, 'r') as f:
        for line in f.readlines():
            music_feat = np.load(pjoin(opt.music_dir, line.strip() + '.npy'))
            music_list.append(music_feat)
            name_list.append(line.strip())

    token_lens = torch.tensor([t_lens for _ in range(len(music_list))]).to(opt.device)
    token_lens = (token_lens // 4) * num_part
    m_length = [t_lens for _ in range(len(music_list))]
    captions = torch.tensor(np.array(music_list)).to(opt.device)

    sample = 0
    converter = Joint2BVHConvertor(model_type='smpl')

    t = time.time()
    with torch.no_grad():
        # mids [1793, 300]
        mids = m2m_transformer.generate(captions, token_lens,
                                        timesteps=28,
                                        cond_scale=4,
                                        temperature=1)
        
        mids = res_model.generate(mids, captions, token_lens, temperature=0.8, cond_scale=5)
 
        pred_motions = vq_model.forward_decoder(mids)
        

        pred_motions = pred_motions.detach().cpu().numpy()

        data = inv_transform(pred_motions)
    
    print('time_elapsed:', time.time() - t)

    t = time.time()
    for k, (caption, joint_data)  in enumerate(zip(name_list, data)):
        print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
        animation_path = pjoin(animation_dir, str(k))
        joint_path = pjoin(joints_dir, str(k))

        os.makedirs(animation_path, exist_ok=True)
        os.makedirs(joint_path, exist_ok=True)

        if nb_joints == 52:
            exp = joint_data[:m_length[k]][:, 623:]
            joint_data = motion_merge(joint_data[:m_length[k]][:, :623])
        else:
            exp = None
            joint_data = joint_data[:m_length[k]]
        np.save(pjoin(motion_feats_dir, caption + '.npy'), joint_data)
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
        np.save(pjoin(joints_all_dir, caption + '.npy'), joint)

        bvh_path = pjoin(animation_path, "sample%d_len%d.bvh" % (k, m_length[k]))
        _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
        convert_to_npz(bvh_path,nb_joints=nb_joints,has_face=False)

        save_path = pjoin(animation_path, "sample%d_len%d.mp4"%(k, m_length[k]))
        audio_path = pjoin(audio_dir, caption + '.wav')
        plot_3d_motion_music(save_path,audio_path, kinematic_chain, joint, title=caption, fps=FPS)
