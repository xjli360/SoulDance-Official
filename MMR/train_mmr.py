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

import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train_mmr", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        print(f"Loading config from {cfg.resume_dir}")
        cfg = read_config(cfg.resume_dir)
        # print(cfg)
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    print(cfg)

    logger.info("Loading the dataloaders")
    
    # train_dataset = instantiate(cfg.data, split="train_tiny")
    # val_dataset = instantiate(cfg.data, split="val_tiny")

    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=True,
    )

    logger.info("Loading the model")
    model = instantiate(cfg.model)

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
