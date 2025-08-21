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

logger = logging.getLogger(__name__)


jukebox_music_feat_path = '/workspace/SoulNet/dataset/souldance/jukebox_feats'   # train + test
save_dir = '/workspace/SoulNet/dataset/souldance/mmr_music_feats/'

@hydra.main(version_base=None, config_path="configs", config_name="text_motion_sim")
def get_mmr_music_feat(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name

    cfg = read_config(run_dir)
    seed_everything(cfg.seed)
    logger.info("Loading the text model")

    cfg.data.text_to_token_emb.path = jukebox_music_feat_path
    
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)
    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    with torch.inference_mode():
        for file in os.listdir(jukebox_music_feat_path):
            file_path = os.path.join(jukebox_music_feat_path, file)
            print(file_path)
            text_x_dict = collate_x_dict([text_model(file_path)])  # 1*150*4800
            lat_t = model.encode(text_x_dict, sample_mean=True)[0] # 1*256
            mmr_music_feat = os.path.join(save_dir, file)
            np.save(mmr_music_feat, lat_t.cpu().numpy())
            print('done!')


if __name__ == "__main__":
    get_mmr_music_feat()
