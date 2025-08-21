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
from omegaconf import DictConfig
import logging
import hydra

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="encode_motion")
def encode_motion(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    npy_path = cfg.npy

    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from src.data.collate import collate_x_dict

    cfg = read_config(run_dir)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    # normalizer = instantiate(cfg.data.motion_loader.normalizer)

    motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    # motion = normalizer(motion)
    motion = motion.to(device)

    motion_x_dict = {"x": motion, "length": len(motion)}

    seed_everything(cfg.seed)
    with torch.inference_mode():
        motion_x_dict = collate_x_dict([motion_x_dict])
        latent = model.encode(motion_x_dict, sample_mean=True)[0]
        latent = latent.cpu().numpy()

    fname = os.path.split(npy_path)[1]
    output_folder = os.path.join(run_dir, "encoded")
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, fname)

    np.save(path, latent)
    logger.info(f"Encoding done, latent saved in:\n{path}")



if __name__ == "__main__":
    encode_motion()
