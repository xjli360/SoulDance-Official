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
import orjson
import json
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

from src.model import TextToEmb


class MusicEmbeddings(ABC):
    def __init__(
        self,
        modelname: str,
        path: str = "",
        device: str = "cpu",
        preload: bool = True,
        disable: bool = False,
    ):
        self.modelname = modelname
        self.base_path = path
        self.cache = {}
        self.device = device
        self.disable = disable
        self.embeddings_index = {}

    def __contains__(self, text):
        return text in self.embeddings_index

    def get_model(self):
        model = getattr(self, "model", None)
        return model

    def __call__(self, path):
        # name = name.split('/')[-1]
        embeddings_folder = path
        # print(embeddings_folder)
        with open(embeddings_folder,'rb') as f:
            embedding = np.load(f)
            torch_embedding=torch.from_numpy(embedding).to(dtype=torch.float, device=self.device)
        x_dict = {"x": torch_embedding, "length": len(torch_embedding)}
        return x_dict
    


