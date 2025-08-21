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
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import ResidualTransformer
from models.mask_transformer.transformer_trainer import ResidualTransformerTrainer
from models.vq.model import RVQVAE,HRVQVAE

from options.train_option import TrainM2MOptions

from utils.plot_script import plot_3d_motion_music
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils import paramUtil

from data.m2m_dataset import Music2MotionDataset
from motion_loaders.dataset_motion_loader import get_music_motion_loader
from models.m2m_eval_wrapper import EvaluatorModelWrapper

music_wav_path = None
motion_fps = 60 

def motion_merge(body_hands):
    # 30 * 3 + 30 * 6 + 30 * 3 = 360
    body = body_hands[..., :263]
    hands = body_hands[..., 263:623]
    motion_623 = np.concatenate((body[:, :4+(22 - 1)*3], 
                                hands[:, :30*3], 
                                body[:, 4+(22 - 1)*3:4+(22 - 1)*3+(22 - 1)*6], 
                                hands[:, 30*3:30*9],
                                body[:, 4+(22 - 1)*9:4+(22 - 1)*9+22*3], 
                                hands[:, 30*9:30*9+30*3], 
                                body[:, -4:]), axis=1)
    return motion_623

def plot_m2m(data, save_dir, captions, m_lengths):
    if opt.dataset_name == "aistpp":
        data = train_dataset.inv_transform(data)
    # audio_dir = '/mnt/bn/code/EDGE/data/all_music_sliced_5'
    audio_dir = music_wav_path
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        if opt.dataset_name == 'souldance':
            joint_data = joint_data[:, :623]
            joint_data = motion_merge(joint_data)
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        audio_path = pjoin(audio_dir, caption + '.wav')
        plot_3d_motion_music(save_path,audio_path, kinematic_chain, joint, title=caption, fps=motion_fps)

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                dim_pose,
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
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.to(opt.device)
    return vq_model, vq_opt

def load_hvq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = HRVQVAE(vq_opt,
                dim_pose,
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
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.vq_name, 'model', 'latest.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.to(opt.device)
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainM2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/res/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 'finedance':
        opt.data_root = './dataset/finedance/'
        opt.motion_dir = pjoin(opt.data_root, 'motion_263')
        opt.mmr_loss = True
        music_wav_path = '/mnt/bn/HumanTOMATO/datasets/finedance_data/music'
        opt.music_dir = '/mnt/bn/MMR/datasets/mmr_finedance_feats' # 256
        opt.joints_num = 52
        radius = 4
        motion_fps = 30
        dim_pose = 623
        opt.max_motion_length = 150

        kinematic_chain = paramUtil.t2m_body_hand_kinematic_chain
        dataset_opt_path = '/mnt/bn/souldance-codes/checkpoints/finedance/opt.txt'

    elif opt.dataset_name == 'souldance': #TODO
        opt.data_root = './dataset/souldance/'
        opt.motion_dir = pjoin(opt.data_root, 'motion_723')
        music_wav_path = '/mnt/bn/HumanTOMATO/datasets/souldance_data/music'

        opt.music_dir = '/mnt/bn/MMR/datasets/mmr_souldance_music_feats' # 256
        opt.joints_num = 52
        radius = 4
        motion_fps = 60
        dim_pose = 723
        opt.max_motion_length = 300

        kinematic_chain = paramUtil.t2m_body_hand_kinematic_chain
        dataset_opt_path = '/mnt/bn/souldance-codes/checkpoints/souldance/opt.txt'

    elif opt.dataset_name == "aistpp":
        opt.data_root = './dataset/aistpp/'
        opt.motion_dir = pjoin(opt.data_root, 'motion_263')
        music_wav_path = '/mnt/bn/HumanTOMATO/datasets/aistpp_data/music'
        opt.music_dir = '/mnt/bn/MMR/datasets/mmr_asitpp_music_feats' # 256 1009

        opt.joints_num = 22
        opt.mmr_loss = True
        radius = 4
        dim_pose = 263
        motion_fps = 60
        opt.max_motion_length = 300

        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/m2m/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')
    vq_model, vq_opt = load_hvq_model()  if opt.hrvq else load_vq_model()

    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    print(opt)

    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='music',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                          # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=opt.share_weight,
                                          clip_version=clip_version,
                                          opt=opt)

    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())

    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    mean = None
    std = None
    if opt.dataset_name == "aistpp":
        mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
        std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_dataset = Music2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Music2MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    eval_val_loader, _ =  get_music_motion_loader(dataset_opt_path, 32, mean, std,'val', device=opt.device)

    # wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    # eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # from MMR.src.config import read_config
    # wrapper_opt = read_config('/mnt/bn/souldance-codes/MMR/outputs/mmr_0926_1/') # humantomato -> humanml3d
    # eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model, eval_wrapper)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_m2m)