import os
import pdb
import os.path as osp
import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from smplx import SMPL,SMPLX
# from utils.smplx2joints import SMPLX
from utils.flame_pytorch.flame import FLAME
from utils.flame_pytorch.config import get_flame_config
import copy

#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)
from MMR.encode_dataset import encode_batch_motion


import sys
sys.path.append('/mnt/bn/MMR')

# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


def motion_merge(body_hands):
    body = body_hands[..., :263]
    hands = body_hands[..., 263:623]
    motion_623 = torch.cat((body[..., :4+(22 - 1)*3], 
                            hands[..., :30*3], 
                            body[..., 4+(22 - 1)*3:4+(22 - 1)*3+(22 - 1)*6], 
                            hands[..., 30*3:30*9],
                            body[..., 4+(22 - 1)*9:4+(22 - 1)*9+22*3], 
                            hands[..., 30*9:30*9+30*3], 
                            body[..., -4:]), dim=-1)
    return motion_623

# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def euclidean_distance_matrix_np(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


@torch.no_grad()
def evaluation_vqvae_face(val_loader,smpl_dir, net):
    net.eval()
    face_gt_list = []
    face_pre_list = []
    for i, batch_data in enumerate(val_loader):

        motion = batch_data.cuda()
        bs, seq = motion.shape[0], motion.shape[1]
        smplx = SMPLX(model_path=smpl_dir, gender='NEUTRAL', batch_size=bs)

        pred_pose_eval, loss_commit, perplexity = net(motion)

        face_gt = smplx.forward(face_shape=motion).face.detach().numpy()
        face_pre = smplx.forward(face_shape=pred_pose_eval).face.detach().numpy()

        face_gt_list.append(face_gt)
        face_pre_list.append(face_pre)

        num_poses += bs * seq

    face_gt_np = np.concatenate(face_gt_list)
    face_pre_np = np.concatenate(face_pre_list)

    facev_l2_score = np.linalg.norm(face_gt_np - face_pre_np)

    print("Face L2:{}".format(facev_l2_score))
    return facev_l2_score


@torch.no_grad()
def evaluation_vqvae_whole(val_loader, net, eval_wrapper,num_joint, model_dir):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []
    nb_sample = 0
    mpjpe,mpjpe_body,mpjpe_hands = 0, 0, 0
    num_poses = 0
    face_vertex_error = 0

    flame_model_path = "/mnt/bn/souldance-codes/utils/flame_pytorch/model/generic_model.pkl"
    static_landmark_embedding_path = '/mnt/bn/souldance-codes/utils/flame_pytorch/model/flame_static_embedding.pkl'
    for i, batch_data in enumerate(val_loader):
        if i > 2:
            break

        motion = batch_data.cuda()
        # conds, music_names, motion, m_lens = batch_data 
        # m_length = m_lens.cuda()
        # conds = conds.cuda()
        # motion = motion.cuda().float()
        bs, seq = motion.shape[0], motion.shape[1]
        m_length = torch.full((bs,), seq).cuda()

        # import pdb;pdb.set_trace()
        pred_pose_eval, loss_commit, perplexity = net(motion)

        # face
        flamelayer = FLAME(flame_model_path, static_landmark_embedding_path, bs*seq).cuda()
        expression_params = torch.zeros(bs*seq, 50, dtype=torch.float32).cuda()
        pose_params = torch.zeros(bs*seq, 6, dtype=torch.float32).cuda()
     
        gt_face, _ = flamelayer(motion[...,623:].reshape(-1, 100).float(), expression_params, pose_params)
        pred_face, _ = flamelayer(pred_pose_eval[...,623:].reshape(-1, 100).float(), expression_params, pose_params)
        face_vertex_error += np.linalg.norm(gt_face.detach().cpu().numpy() - pred_face.detach().cpu().numpy())
        del expression_params, pose_params, gt_face, pred_face,flamelayer

        # motion
        gt_body, pred_body = motion[...,:263], pred_pose_eval[...,:263]
        bgt_latents = eval_wrapper.get_motion_embeddings(gt_body, m_length).detach().cpu().numpy()
        bpred_latents = eval_wrapper.get_motion_embeddings(pred_body, m_length).detach().cpu().numpy()

        motion_pred_list.append(bgt_latents)
        motion_annotation_list.append(bpred_latents)

        # (256, 64, 263)
        # bgt = motion.detach().cpu().numpy()
        # bpred = pred_pose_eval.detach().cpu().numpy()

        # pdb.set_trace()
        # gt  256, 64, 22, 3
        # bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        # gt = recover_from_ric(torch.from_numpy(bgt).float(), num_joint)
        # pred = recover_from_ric(torch.from_numpy(bpred).float(), num_joint)

        # motion
        gt_motion_263 = val_loader.dataset.motion_merge(motion[...,:623])
        pred_motion_263 = val_loader.dataset.motion_merge(pred_pose_eval[...,:623])

        gt = recover_from_ric(gt_motion_263.detach().float(), num_joint)
        pred = recover_from_ric(pred_motion_263.detach().float(), num_joint)
        # import pdb;pdb.set_trace()
        mpjpe += torch.sum(calculate_mpjpe(gt.reshape(-1, num_joint, 3), pred.reshape(-1, num_joint, 3)))

        gt_body, gt_hands = gt[:, :22], gt[:, 22:52]
        pred_body, pred_hands = pred[:, :22], pred[:, 22:52]

        mpjpe_body += torch.sum(calculate_mpjpe(gt_body.reshape(-1, 22, 3), pred_body.reshape(-1, 22, 3)))
        mpjpe_hands += torch.sum(calculate_mpjpe(gt_hands.reshape(-1, 30, 3), pred_hands.reshape(-1, 30, 3)))

        num_poses += bs * seq

        torch.cuda.empty_cache()

    motion_annotation_np = np.concatenate(motion_annotation_list)
    motion_pred_np = np.concatenate(motion_pred_list)


    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    mpjpe = mpjpe / num_poses
    mpjpe_body = mpjpe_body / num_poses
    mpjpe_hands = mpjpe_hands / num_poses
    face_vertex_error = face_vertex_error / num_poses
    msg = "FID:{} DIV_pred:{} DIV_real:{} MPJPE:{} MPJPE_body:{} MPJPE_hands:{} Face_Vertex_Error:{}".format(fid, diversity_pred, diversity_real, mpjpe, mpjpe_body, mpjpe_hands, face_vertex_error)
    print(msg)
    log_file = os.path.join(model_dir, 'log.txt')
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

    return mpjpe



@torch.no_grad()
def evaluation_vqvae_mpjpe(val_loader, net, eval_wrapper,num_joint, model_dir):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []
    nb_sample = 0
    mpjpe = 0
    num_poses = 0
    for i, batch_data in enumerate(val_loader):
        if i > 100:
            break

        motion = batch_data.cuda()
        # conds, music_names, motion, m_lens = batch_data 
        # m_length = m_lens.cuda()
        # conds = conds.cuda()
        # motion = motion.cuda().float()
        bs, seq = motion.shape[0], motion.shape[1]
        m_length = torch.full((bs,), seq).cuda()

        # import pdb;pdb.set_trace()
        pred_pose_eval, loss_commit, perplexity = net(motion)

        # bgt = val_loader.dataset.inv_transform(motion)
        # bpred = val_loader.dataset.inv_transform(pred_pose_eval)
        # bgt_latents = eval_wrapper.get_motion_embeddings(bgt).detach().cpu().numpy() # erro
        # bpred_latents = eval_wrapper.get_motion_embeddings(bpred).detach().cpu().numpy()
        bgt_latents = eval_wrapper.get_motion_embeddings(motion, m_length).detach().cpu().numpy()
        bpred_latents = eval_wrapper.get_motion_embeddings(pred_pose_eval, m_length).detach().cpu().numpy()


        motion_pred_list.append(bgt_latents)
        motion_annotation_list.append(bpred_latents)

        # (256, 64, 263)
        # bgt = motion.detach().cpu().numpy()
        # bpred = pred_pose_eval.detach().cpu().numpy()

        # pdb.set_trace()
        # gt  256, 64, 22, 3
        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())

        gt = recover_from_ric(torch.from_numpy(bgt).float(), num_joint)
        pred = recover_from_ric(torch.from_numpy(bpred).float(), num_joint)

        # import pdb;pdb.set_trace()
        mpjpe += torch.sum(calculate_mpjpe(gt.reshape(-1, 22, 3), pred.reshape(-1, 22, 3)))
        num_poses += bs * seq

    motion_annotation_np = np.concatenate(motion_annotation_list)
    motion_pred_np = np.concatenate(motion_pred_list)


    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    mpjpe = mpjpe / num_poses
    # print("FID:{} DIV_pred:{} DIV_real:{} MPJPE:{}".format(fid, diversity_pred, diversity_real, mpjpe))
    msg = "FID:{} DIV_pred:{} DIV_real:{} MPJPE:{}".format(fid, diversity_pred, diversity_real, mpjpe)
    print(msg)
    log_file = os.path.join(model_dir, 'log.txt')
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    return mpjpe


def motion_div_x(motion,x=0.0):
    motion[...,0:1] = motion[...,0:1] * x
    motion[...,1:3] = motion[...,1:3] * x
    motion[...,3:4] = motion[...,3:4] * x
    motion[...,4 + (22 - 1) * 9 + 22 * 3:] = motion[...,4 + (22 - 1) * 9 + 22 * 3:] * x
    motion[4 + (22 - 1) * 9: 4 + (22 - 1) * 9 + 3] = motion[4 + (22 - 1) * 9: 4 + (22 - 1) * 9 + 3] * x
    return motion


@torch.no_grad()
def evaluation_mask_transformer(opt, val_loader, trans, vq_model, writer, ep, best_fid,eval_wrapper, plot_func, save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    music_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []

    mmr_motion_annotation_list = []
    mmr_motion_pred_list = []

    time_steps = 18
    cond_scale = 4 # 2
    best_mmr_score = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    if opt.hrvq:
        print('Use HRVQ !!!')
        part_num = 3
    else:
        part_num = 1

    nb_sample = 0
    # for i in range(1):
    for i, batch in enumerate(val_loader):
        conds, music_names, motion, m_lens = batch  # conds == mmr music encoder
        m_length = m_lens.cuda()
        conds = conds.cuda()
        pose = motion.cuda().float()
        bs, seq = motion.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        if i < 3:
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(conds, (m_length//4) * part_num, time_steps, cond_scale, temperature=1) # mids torch.Size([32, 74*3, 1])

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                # mids = mids.repeat(1,3,1)
                pred_motions = vq_model.forward_decoder(mids) # (b, m_len, 263)
                # import pdb;pdb.set_trace()
                em_pred = eval_wrapper.get_motion_embeddings(pred_motions[...,:263],m_length).unsqueeze(1).detach().cpu().numpy()

                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred)
            motion_multimodality_batch = np.concatenate(motion_multimodality_batch, axis=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)

        # (b, seqlen)
        # import pdb;pdb.set_trace()
        mids = trans.generate(conds, (m_length//4) * part_num, time_steps, cond_scale, temperature=1)
        # import pdb;pdb.set_trace()
        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)

        pred_motions = vq_model.forward_decoder(mids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions[..., :263],m_length).detach().cpu().numpy() 
        em = eval_wrapper.get_motion_embeddings(pose[..., :263],m_length).detach().cpu().numpy()

        mmr_em_pred = eval_wrapper.get_mmr_motion_embeddings(pred_motions[..., :263]).detach().cpu().numpy()
        mmr_em = eval_wrapper.get_mmr_motion_embeddings(pose[..., :263]).detach().cpu().numpy()

        mu = conds.cpu().numpy()   # == conds.cpu().numpy()
        # mu = eval_wrapper.get_music_embeddings_from_path(music_names)
        motion_annotation_list.append(em)
        music_annotation_list.append(mu)
        motion_pred_list.append(em_pred)

        mmr_motion_annotation_list.append(mmr_em)
        mmr_motion_pred_list.append(mmr_em_pred)

        nb_sample += bs

    motion_annotation_np = np.concatenate(motion_annotation_list)
    music_annotation_np = np.concatenate(music_annotation_list)
    motion_pred_np = np.concatenate(motion_pred_list)
    motion_multimodality_np = np.concatenate(motion_multimodality)

    mmr_motion_annotation_np = np.concatenate(mmr_motion_annotation_list)
    mmr_motion_pred_np = np.concatenate(mmr_motion_pred_list)

    # mmr_score_list_gt = get_score_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(motion_annotation_np)).cpu().tolist()
    # mmr_score_list_pre = get_score_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(motion_pred_np)).cpu().tolist()

    mmr_score_list_gt = euclidean_distance_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(mmr_motion_annotation_np)).cpu().tolist()
    mmr_score_list_pre = euclidean_distance_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(mmr_motion_pred_np)).cpu().tolist()
    mmr_score_avg_gt = np.sum(np.diagonal(np.array(mmr_score_list_gt))) / len(mmr_score_list_gt) / 100.0
    mmr_score_avg_pre = np.sum(np.diagonal(np.array(mmr_score_list_pre))) / len(mmr_score_list_pre) / 100.0

    # import pdb;pdb.set_trace()
    multimodality = calculate_multimodality(motion_multimodality_np, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f} , MM. {multimodality:.4f}, MMR Matching Pre Score. {mmr_score_avg_pre:.4f}, MMR Matching Gt Score. {mmr_score_avg_gt:.4f} \n"
    print(msg)
    log_file = os.path.join(opt.save_root, 'log.txt')
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(opt.save_root, 'model', 'net_best_fid.tar'), ep)
    best_mmr_score = max(best_mmr_score, mmr_score_avg_pre)
    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [music_names[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(opt.save_root, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_mmr_score, writer


@torch.no_grad()
def evaluation_res_transformer(opt, val_loader, trans, vq_model, writer, ep, best_fid,eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    music_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []

    mmr_motion_annotation_list = []
    mmr_motion_pred_list = []

    time_steps = 18
    cond_scale = 4 # 2
    best_mmr_score = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    if opt.hrvq:
        print('Use HRVQ !!!')
        part_num = 3
    else:
        part_num = 1

    nb_sample = 0
    # for i in range(1):
    for i, batch_data in enumerate(val_loader):

        # if i > 200:
        #     break
        conds, music_names, motion, m_lens = batch_data
        m_length = m_lens.cuda()
        pose = motion.cuda().float()
        conds = conds.cuda()

        bs, seq = motion.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        code_indices, all_codes = vq_model.encode(pose)
        if i < 3:
            motion_multimodality_batch = []
            for _ in range(30):
                if ep == 0:
                    pred_ids = code_indices[..., 0:1]
                else:
                    pred_ids = trans.generate(code_indices[..., 0], conds, (m_length//4) * part_num, temperature=temperature, cond_scale=cond_scale)
                pred_motions = vq_model.forward_decoder(pred_ids) # (b, m_len, 263)
                # import pdb;pdb.set_trace()
                em_pred = eval_wrapper.get_motion_embeddings(pred_motions[...,:263],m_length).unsqueeze(1).detach().cpu().numpy()

                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred)
            motion_multimodality_batch = np.concatenate(motion_multimodality_batch, axis=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)

        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], conds, (m_length//4) * part_num, temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        # em_pred = eval_wrapper.get_motion_embeddings(pred_motions.detach().cpu().numpy())
        # em = eval_wrapper.get_motion_embeddings(pose.detach().cpu().numpy())

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions[..., :263],m_length).detach().cpu().numpy() 
        em = eval_wrapper.get_motion_embeddings(pose[..., :263],m_length).detach().cpu().numpy()

        mmr_em_pred = eval_wrapper.get_mmr_motion_embeddings(pred_motions[..., :263]).detach().cpu().numpy()
        mmr_em = eval_wrapper.get_mmr_motion_embeddings(pose[..., :263]).detach().cpu().numpy()

        mu = conds.cpu().numpy()   # == conds.cpu().numpy()
        # mu = eval_wrapper.get_music_embeddings_from_path(music_names)
        motion_annotation_list.append(em)
        music_annotation_list.append(mu)
        motion_pred_list.append(em_pred)

        mmr_motion_annotation_list.append(mmr_em)
        mmr_motion_pred_list.append(mmr_em_pred)

        nb_sample += bs

    motion_annotation_np = np.concatenate(motion_annotation_list)
    music_annotation_np = np.concatenate(music_annotation_list)
    motion_pred_np = np.concatenate(motion_pred_list)
    motion_multimodality_np = np.concatenate(motion_multimodality)

    mmr_motion_annotation_np = np.concatenate(mmr_motion_annotation_list)
    mmr_motion_pred_np = np.concatenate(mmr_motion_pred_list)

    # mmr_score_list_gt = get_score_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(motion_annotation_np)).cpu().tolist()
    # mmr_score_list_pre = get_score_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(motion_pred_np)).cpu().tolist()

    mmr_score_list_gt = euclidean_distance_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(mmr_motion_annotation_np)).cpu().tolist()
    mmr_score_list_pre = euclidean_distance_matrix(torch.from_numpy(music_annotation_np), torch.from_numpy(mmr_motion_pred_np)).cpu().tolist()
    mmr_score_avg_gt = np.sum(np.diagonal(np.array(mmr_score_list_gt))) / len(mmr_score_list_gt) / 100.0
    mmr_score_avg_pre = np.sum(np.diagonal(np.array(mmr_score_list_pre))) / len(mmr_score_list_pre) / 100.0

    # import pdb;pdb.set_trace()
    multimodality = calculate_multimodality(motion_multimodality_np, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f} , MM. {multimodality:.4f}, MMR Matching Pre Score. {mmr_score_avg_pre:.4f}, MMR Matching Gt Score. {mmr_score_avg_gt:.4f} \n"
    print(msg)
    log_file = os.path.join(opt.save_root, 'log.txt')
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(opt.save_root, 'model', 'net_best_fid.tar'), ep)
    best_mmr_score = max(best_mmr_score, mmr_score_avg_pre)
    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [music_names[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(opt.save_root, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_mmr_score, writer




@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality

