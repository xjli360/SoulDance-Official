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
import numpy as np
import torch
sys.path.append('/mnt/bn/souldance-codes')
sys.path.append('/mnt/bn/MMR')

from utils.get_opt import get_opt
from models.m2m_eval_wrapper import EvaluatorModelWrapper
from utils.metrics import *



dataset_opt_path = '/mnt/bn/souldance-codes/checkpoints/souldance/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


def cal_fid(pre_motion_263_dir, gt_motion_263_dir):

    motion_annotation_list = []
    music_annotation_list = []
    motion_pred_list = []

    for m_file in os.listdir(pre_motion_263_dir):
        pre_motion = np.load(os.path.join(pre_motion_263_dir, m_file))
        gt_motion = np.load(os.path.join(gt_motion_263_dir, m_file))
        muisc_feat = np.load(os.path.join(music_feats_dir, m_file))

        motion_annotation_list.append(gt_motion)
        motion_pred_list.append(pre_motion)
        music_annotation_list.append(muisc_feat)

    motion_gt = torch.from_numpy(np.array(motion_annotation_list))
    motion_pre = torch.from_numpy(np.array(motion_pred_list))
    music_mmr = np.array(music_annotation_list)

    m_length_gt = torch.from_numpy(np.full((motion_gt.shape[0],), motion_gt.shape[1]))
    m_length_pre = torch.from_numpy(np.full((motion_pre.shape[0],), motion_pre.shape[1]))

    # import pdb;pdb.set_trace()
    l_pred =  eval_wrapper.get_motion_embeddings(motion_pre[...,:263].to(torch.float32),m_length_pre).detach().cpu().numpy()  # 623的[...,:263]!=723的[...,:263]
    l_gt = eval_wrapper.get_motion_embeddings(motion_gt[...,:263].to(torch.float32),m_length_gt).detach().cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(l_gt)
    mu, cov = calculate_activation_statistics(l_pred)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity = calculate_diversity(l_pred, 100)

    mmr_em_pred = eval_wrapper.get_mmr_motion_embeddings(motion_pre[..., :263].to(torch.float32)).detach().cpu().numpy()
    mmr_em = eval_wrapper.get_mmr_motion_embeddings(motion_gt[..., :263].to(torch.float32)).detach().cpu().numpy()
    mmr_score_list_gt = euclidean_distance_matrix(music_mmr, mmr_em).tolist()
    mmr_score_list_pre = euclidean_distance_matrix(music_mmr, mmr_em_pred).tolist()
    mmr_score_avg_gt = np.sum(np.diagonal(np.array(mmr_score_list_gt))) / len(mmr_score_list_gt) / 100.0
    mmr_score_avg_pre = np.sum(np.diagonal(np.array(mmr_score_list_pre))) / len(mmr_score_list_pre) / 100.0

    print('fid:{}, div:{}, mmr_gt:{}, mmr_pre:{}'.format(fid, diversity, mmr_score_avg_gt, mmr_score_avg_pre))



def cal_fid_623(pre_motion_263_dir, gt_motion_263_dir):
    motion_annotation_list = []
    music_annotation_list = []
    motion_pred_list = []

    for m_file in os.listdir(pre_motion_263_dir):
        pre_motion = np.load(os.path.join(pre_motion_263_dir, m_file))
        gt_motion = np.load(os.path.join(gt_motion_263_dir, m_file))
        muisc_feat = np.load(os.path.join(music_feats_dir, m_file))

        motion_annotation_list.append(gt_motion)
        motion_pred_list.append(pre_motion)
        music_annotation_list.append(muisc_feat)

    motion_gt = torch.from_numpy(np.array(motion_annotation_list))
    motion_pre = torch.from_numpy(np.array(motion_pred_list))
    music_mmr = np.array(music_annotation_list)


    m_length_gt = torch.from_numpy(np.full((motion_gt.shape[0],), motion_gt.shape[1]))
    m_length_pre = torch.from_numpy(np.full((motion_pre.shape[0],), motion_pre.shape[1]))
    #
    # m_length_gt = torch.from_numpy(np.full((motion_gt.shape[0],), motion_gt.shape[1]-100))
    # m_length_pre = torch.from_numpy(np.full((motion_pre.shape[0],), motion_pre.shape[1]-100))

    data_263 = torch.cat((motion_pre[..., :4+(22 - 1)*3], motion_pre[..., 4+(52 - 1)*3:4+(52 - 1)*3+(22 - 1)*6], motion_pre[..., 4 + (52 - 1)*9: 4 + (52 - 1) *9 + 22 *3], motion_pre[..., -4:]), dim=-1)
    print(data_263.shape)
    # l_pred =  eval_wrapper.get_motion_embeddings(data_263[...,:263].to(torch.float32),m_length_pre).detach().cpu().numpy()  # 623的[...,:263]!=723的[...,:263]
    # l_gt = eval_wrapper.get_motion_embeddings(motion_gt[...,:263].to(torch.float32),m_length_gt).detach().cpu().numpy()

    l_pred =  eval_wrapper.get_motion_embeddings(data_263[:,100:,:263].to(torch.float32),m_length_pre).detach().cpu().numpy()  # 623的[...,:263]!=723的[...,:263]
    l_gt = eval_wrapper.get_motion_embeddings(motion_gt[:,100:,:263].to(torch.float32),m_length_gt).detach().cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(l_gt)
    mu, cov = calculate_activation_statistics(l_pred)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity = calculate_diversity(l_pred, 100)

    print('gt diversity:', calculate_diversity(l_gt, 100))

    # Calculate MMR-Matching Scores using the new function
    motion_gt_sequences = [motion_gt[i, 100:, :263].to(torch.float32) for i in range(len(motion_annotation_list))]
    motion_pre_sequences = [data_263[i, 100:, :263].to(torch.float32) for i in range(len(motion_annotation_list))]
    
    mmr_score_avg_gt = calculate_mmr_matching_score(motion_gt_sequences, music_mmr, eval_wrapper)
    mmr_score_avg_pre = calculate_mmr_matching_score(motion_pre_sequences, music_mmr, eval_wrapper)

    print('fid:{}, div:{}, mmr_gt:{}, mmr_pre:{}'.format(fid, diversity, mmr_score_avg_gt, mmr_score_avg_pre))


def calculate_mmr_matching_score(motion_sequences, music_sequences, eval_wrapper, fps=30, mu_weight=0.7, lambda_weight=0.3):
    """
    Calculate MMR-Matching Score for given motion and music sequences.
    """
    segment_length = fps
    mmr_score_list = []
    
    for i in range(len(motion_sequences)):
        motion_seq = motion_sequences[i]
        music_seq = music_sequences[i]
        
        seq_length = motion_seq.shape[0]
        num_segments = seq_length // segment_length

        if num_segments < 2: 
            continue
            
        # Get embeddings for each segment
        z_segments = []
        m_segments = []
        
        for t in range(num_segments):
            start_idx = t * segment_length
            end_idx = (t + 1) * segment_length
            motion_seg = motion_seq[start_idx:end_idx].unsqueeze(0)
            z = eval_wrapper.get_mmr_motion_embeddings(motion_seg).detach().cpu().numpy().squeeze()
            z_segments.append(z)
            
            if len(music_seq.shape) > 1 and music_seq.shape[0] >= num_segments:
                m_segments.append(music_seq[t])
            else:
                m_segments.append(music_seq)
        
        z_segments = np.array(z_segments)
        m_segments = np.array(m_segments)
        
        # Calculate MMR-MS
        # First term: feature space distance
        feature_dist = np.sum((z_segments - m_segments) ** 2, axis=1)
        
        # Second term: feature dynamics distance
        if num_segments > 1:
            delta_z = z_segments[1:] - z_segments[:-1]
            delta_m = m_segments[1:] - m_segments[:-1] if len(m_segments) > 1 else np.zeros_like(delta_z)
            dynamics_dist = np.sum(np.linalg.norm(delta_z - delta_m, axis=1))
        else:
            dynamics_dist = 0
        
        # MMR-MS with μ and λ weights
        mmr_ms = np.sqrt(mu_weight * np.mean(feature_dist) + lambda_weight * dynamics_dist)
        mmr_score_list.append(mmr_ms)
    
    return np.mean(mmr_score_list) if mmr_score_list else 0

def cal_mutilmodality(pre_motion_dir_list):
    motion_multimodality = []
    for pre_motion_dir in pre_motion_dir_list:
        motion_pred_list = []
        for m_file in os.listdir(pre_motion_dir):
            motion_263 = np.load(os.path.join(pre_motion_dir, m_file))
            motion_pred_list.append(motion_263)
        motion_pre = torch.from_numpy(np.array(motion_pred_list))
        m_length_pre = torch.from_numpy(np.full((motion_pre.shape[0],), motion_pre.shape[1]))
        l_pred =  eval_wrapper.get_motion_embeddings(motion_pre[...,:263],m_length_pre).detach().cpu().numpy()
        motion_multimodality.append(l_pred)

    motion_multimodality_np = np.concatenate(motion_multimodality).reshape(-1, l_pred.shape[0], 512)
    multimodality = calculate_multimodality(motion_multimodality_np, 100)
    print(multimodality)


def cal_mutilmodality_623(pre_motion_dir_list):
    motion_multimodality = []
    for pre_motion_dir in pre_motion_dir_list:
        motion_pred_list = []
        for m_file in os.listdir(pre_motion_dir):
            motion_263 = np.load(os.path.join(pre_motion_dir, m_file))
            motion_pred_list.append(motion_263)
        motion_pre = torch.from_numpy(np.array(motion_pred_list))
        motion_pre = torch.cat((motion_pre[..., :4+(22 - 1)*3], motion_pre[..., 4+(52 - 1)*3:4+(52 - 1)*3+(22 - 1)*6], motion_pre[..., 4 + (52 - 1)*9: 4 + (52 - 1) *9 + 22 *3], motion_pre[..., -4:]), dim=-1)
        print(motion_pre.shape)
        m_length_pre = torch.from_numpy(np.full((motion_pre.shape[0],), motion_pre.shape[1]))
        l_pred =  eval_wrapper.get_motion_embeddings(motion_pre[...,:263],m_length_pre).detach().cpu().numpy()
        motion_multimodality.append(l_pred)

    # import pdb;pdb.set_trace()
    motion_multimodality_np = np.concatenate(motion_multimodality).reshape(-1, l_pred.shape[0], 512)
    multimodality = calculate_multimodality(motion_multimodality_np, 100)
    print(multimodality)
    
    


if __name__ == '__main__':
    # soulnet-souldance rvq
    # pre_motion_263_dir = '/mnt/bn/souldance-codes/generation/exp_souldance_rvq5_1024/motion_feats' 
    # gt_motion_263_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_263'
    # music_feats_dir = '/mnt/bn/MMR/datasets/mmr_souldance_music_feats' 
    # cal_fid_623(pre_motion_263_dir, gt_motion_263_dir)

    # soulnet-souldance hrvq
    pre_motion_263_dir = '/mnt/bn/souldance-codes/generation/exp_souldance_hrvq5/motion_feats' 
    gt_motion_263_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_263'
    music_feats_dir = '/mnt/bn/MMR/datasets/mmr_souldance_music_feats' 
    cal_fid_623(pre_motion_263_dir, gt_motion_263_dir)



