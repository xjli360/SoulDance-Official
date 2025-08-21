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

import sys
import os
from os.path import join as pjoin


import torch
from torch.utils.data import DataLoader
from models.vq.model import RVQVAE,HRVQVAE
from options.vq_option import arg_parse
import utils.eval_t2m as eval_t2m
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from data.m2m_dataset import DanceDataset

from utils.eval_m2m import evaluation_vqvae_mpjpe, evaluation_vqvae_whole
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor
from visualization.utils.emage_npz import convert_to_npz

def load_vq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location=vq_opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

def load_hvq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = HRVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location=vq_opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

if __name__ == "__main__":
    ##### ---- Exp dirs ---- #####
    args = arg_parse(False)
    args.device = torch.device('cuda:0')

    ##### ---- Dataloader ---- #####
    args.nb_joints = 52
    args.data_root = './dataset/souldance/'
    args.motion_dir = pjoin(args.data_root, 'motion_723')
    args.dataset_name = "souldance"
    args.batch_size = 1

    args.joints_num = 52
    fps = 60
    args.max_motion_length = 300
    dim_pose = 723
    val_split_file = pjoin(args.data_root, 'val_tiny.txt')
    val_dataset = DanceDataset(args, val_split_file)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=20,
                            shuffle=False, pin_memory=True)
    
    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name,'vqs_eval_1108')
    os.makedirs(args.out_dir, exist_ok=True)
    print(args.out_dir)

    ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, 'rvq0_souldance_512', 'opt.txt')
    rvq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, 'rvq5_souldance', 'opt.txt')
    hrvq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, 'hrvq7_souldance', 'opt.txt')
 
    # net = load_vq_model()
    vq_net, ep = load_vq_model(get_opt(vq_opt_path, device=args.device), 'latest.tar')
    rvq_net, ep = load_vq_model(get_opt(rvq_opt_path, device=args.device), 'latest.tar')
    hrvq_net, ep = load_hvq_model(get_opt(hrvq_opt_path, device=args.device), 'latest.tar')

    vq_net.eval()
    rvq_net.eval()
    hrvq_net.eval()

    vq_net.to('cuda:0')
    rvq_net.to('cuda:0')
    hrvq_net.to('cuda:0')

    converter = Joint2BVHConvertor(model_type='smplx')

    gt_motion_list = []
    gt_face_list = []

    vq_pred_motion_list = []
    rvq_pred_motion_list = []
    hrvq_pred_motion_list = []

    vq_pred_face_list = []
    rvq_pred_face_list = []
    hrvq_pred_face_list = []
    for i, batch_data in enumerate(val_dataset.data):

        motion = torch.from_numpy(batch_data).cuda()
        motion = motion.unsqueeze(0)
        # conds, music_names, motion, m_lens = batch_data 
        # m_length = m_lens.cuda()
        # conds = conds.cuda()
        # motion = motion.cuda().float()
        bs, seq = motion.shape[0], motion.shape[1]
        m_length = torch.full((bs,), seq).cuda()

        vq_pred_pose, _, _ = vq_net(motion)
        rvq_pred_pose, _, _ = rvq_net(motion)
        hrvq_pred_pose, _, _ = hrvq_net(motion)

        # face
        gt_face = motion[...,623:].detach().cpu().numpy()

        vq_pred_face = vq_pred_pose[...,623:].detach().cpu().numpy()
        rvq_pred_face = rvq_pred_pose[...,623:].detach().cpu().numpy()
        hrvq_pred_face = hrvq_pred_pose[...,623:].detach().cpu().numpy()

        gt_face_list.append(gt_face)
        vq_pred_face_list.append(vq_pred_face)
        rvq_pred_face_list.append(rvq_pred_face)
        hrvq_pred_face_list.append(hrvq_pred_face)

        # motion
        gt_motion_263 = val_loader.dataset.motion_merge(motion[...,:623])
        gt = recover_from_ric(gt_motion_263.detach().float(), args.nb_joints).detach().cpu().numpy()

        vq_pred = recover_from_ric(val_loader.dataset.motion_merge(vq_pred_pose[...,:623]).detach().float(), args.nb_joints).detach().cpu().numpy()
        rvq_pred = recover_from_ric(val_loader.dataset.motion_merge(rvq_pred_pose[...,:623]).detach().float(), args.nb_joints).detach().cpu().numpy()
        hrvq_pred = recover_from_ric(val_loader.dataset.motion_merge(hrvq_pred_pose[...,:623]).detach().float(), args.nb_joints).detach().cpu().numpy()

        gt_motion_list.append(gt)
        vq_pred_motion_list.append(vq_pred)
        rvq_pred_motion_list.append(rvq_pred)
        hrvq_pred_motion_list.append(hrvq_pred)

    bvh_path = pjoin(args.out_dir, "vqs_.bvh")
    val_time = 0

    gt_motion_np = np.concatenate(gt_motion_list, axis=1)
    gt_face_np = np.concatenate(gt_face_list, axis=1)

    vq_pred_motion_np = np.concatenate(vq_pred_motion_list, axis=1)
    rvq_pred_motion_np = np.concatenate(rvq_pred_motion_list, axis=1)
    hrvq_pred_motion_np = np.concatenate(hrvq_pred_motion_list, axis=1)

    vq_pred_face_np = np.concatenate(vq_pred_face_list, axis=1)
    rvq_pred_face_np = np.concatenate(rvq_pred_face_list, axis=1)
    hrvq_pred_face_np = np.concatenate(hrvq_pred_face_list, axis=1)

    _, gt_joint = converter.convert(gt_motion_np[0], filename=pjoin(args.out_dir, "gt_{}.bvh".format(val_time)), iterations=100, foot_ik=False,FPS=60)
    convert_to_npz(pjoin(args.out_dir, "gt_{}.bvh".format(val_time)),face_data=gt_face_np[0],nb_joints=52)

    _, vq_joint = converter.convert(vq_pred_motion_np[0], filename=pjoin(args.out_dir, "vq_{}.bvh".format(val_time)), iterations=100, foot_ik=False,FPS=60)
    _, rvq_joint = converter.convert(rvq_pred_motion_np[0], filename=pjoin(args.out_dir, "rvq_{}.bvh".format(val_time)), iterations=100, foot_ik=False,FPS=60)
    _, hrvq_joint = converter.convert(hrvq_pred_motion_np[0], filename=pjoin(args.out_dir, "hrvq_{}.bvh".format(val_time)), iterations=100, foot_ik=False,FPS=60)

    convert_to_npz(pjoin(args.out_dir, "vq_{}.bvh".format(val_time)),face_data=vq_pred_face_np[0],nb_joints=52)  
    convert_to_npz(pjoin(args.out_dir, "rvq_{}.bvh".format(val_time)),face_data=rvq_pred_face_np[0],nb_joints=52)
    convert_to_npz(pjoin(args.out_dir, "hrvq_{}.bvh".format(val_time)),face_data=hrvq_pred_face_np[0],nb_joints=52)







        



    

