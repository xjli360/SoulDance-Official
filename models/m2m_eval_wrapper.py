
# Copyright (c) 2023 Chuan Guo

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


from models.t2m_eval_modules import *
from utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin

from MMR.src.load import load_model_from_cfg
from MMR.src.data.collate import collate_x_dict
from MMR.src.config import read_config

from pytorch_lightning import seed_everything
import numpy as np
import torch
import os


def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    
    mmr_opt = read_config('/mnt/bn/souldance-codes/MMR/outputs/mmr_1009_1/')
    seed_everything(mmr_opt.seed, verbose=False)
    mmr_enc = load_model_from_cfg(mmr_opt,'last', eval_mode=True, device=opt.device) # motion or music

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'm2m', 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc, mmr_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        opt.dim_pose = 263
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.text_encoder, self.motion_encoder, self.movement_encoder, self.mmr_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)
        self.mmr_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        self.mmr_encoder.eval()
        

    # Please note that the results does not follow the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not follow the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

    def buid_mmr_data(self, motion_data):
        mmr_data = {}
        # if not isinstance(motion_data, torch.Tensor):
        #     motion_data = torch.from_numpy(motion_data).to(self.device)
        mmr_data['x'] = motion_data.to(self.device)
        mmr_data['length'] = [motion_data.shape[1] for _ in range(motion_data.shape[0])] 
        mmr_data['mask'] = torch.ones((motion_data.shape[0], motion_data.shape[1]),dtype=torch.bool).to(self.device)
        return mmr_data
    
    # Please note that the results does not follow the order of inputs
    def get_mmr_motion_embeddings(self, motion_x):
        """
        (b,seq_len,latent_dim)
        """
        all_latents = []
        motion_x_dict = self.buid_mmr_data(motion_x)
        # with torch.inference_mode():
        latents = self.mmr_encoder.encode(motion_x_dict, sample_mean=True)
        all_latents.append(latents)
        # latents = np.concatenate(all_latents)
        all_latents = torch.cat(all_latents, dim=0) 
        return all_latents