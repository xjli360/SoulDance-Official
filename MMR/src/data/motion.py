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


class AMASSMotionLoader:
    def __init__(
        self, base_dir, fps, normalizer=None, disable: bool = False, nfeats=None
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.normalizer = normalizer
        self.disable = disable
        self.nfeats = nfeats

    def __call__(self, path, start, end):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        begin = int(start * self.fps)
        end = int(end * self.fps)
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            # print(motion_path)
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            if self.normalizer is not None:
                motion = self.normalizer(motion)
            self.motions[path] = motion

        motion = self.motions[path][begin:end]
        x_dict = {"x": motion, "length": len(motion)}
        return x_dict



class AistMotionLoader:
    def __init__(
        self, base_dir, fps, normalizer=None, disable: bool = False, nfeats=None
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.normalizer = normalizer
        self.disable = disable
        self.nfeats = nfeats

    def __call__(self, path):

        if path not in self.motions:
            name = path.split("/")[-1]
            if name[0] == 'M':
                # motion_path = os.path.join('datasets/motions/guoh3dfeats_m/M/aist_mmr_data', name)
                motion_path = os.path.join('/mnt/bn/souldance-codes/dataset/aistpp/vector_263', name)
            else:
                # motion_path = os.path.join(self.base_dir, name)
                motion_path = os.path.join('/mnt/bn/souldance-codes/dataset/aistpp/vector_263', name)
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            if self.normalizer is not None:
                motion = self.normalizer(motion)
            self.motions[path] = motion

        motion = self.motions[path]
        x_dict = {"x": motion, "length": len(motion)}
        return x_dict


class MMRMotionLoader:
    def __init__(
        self, base_dir, fps, normalizer=None, disable: bool = False, nfeats=None
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.normalizer = normalizer
        self.disable = disable
        self.nfeats = nfeats
        self.eps = 1e-12
        self.finedance_mean = torch.from_numpy(np.load('/mnt/bn/HumanTOMATO/datasets/finedance_data/Mean.npy'))
        self.finedance_std = torch.from_numpy(np.load('/mnt/bn/HumanTOMATO/datasets/finedance_data/Std.npy'))

        self.souldance_mean = torch.from_numpy(np.load('/mnt/bn/HumanTOMATO/datasets/souldance_data/Mean.npy'))
        self.souldance_std = torch.from_numpy(np.load('/mnt/bn/HumanTOMATO/datasets/souldance_data/Std.npy'))

        self.aistpp_mean = torch.from_numpy(np.load('/mnt/bn/HumanTOMATO/datasets/aistpp_data/Mean.npy'))
        self.aistpp_std = torch.from_numpy(np.load('/mnt/bn/HumanTOMATO/datasets/aistpp_data/Std.npy'))

    def __call__(self, path):

        if path not in self.motions:
            name = path.split("/")[-1]
            dataset_name = path.split("/")[-3]
            ## import pdb;pdb.set_trace()
            if dataset_name == 'finedance_data':
                motion_path = os.path.join('/mnt/bn/HumanTOMATO/datasets/finedance_data/motion_263', name)
            elif dataset_name == 'aistpp_data':
                motion_path = os.path.join('/mnt/bn/HumanTOMATO/datasets/aistpp_data/motion_263', name)
            elif dataset_name == 'souldance_data':
                motion_path = os.path.join('/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_263', name)
            else:
                print('{} not exit!'.format(name))
            # print(motion_path)
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            # if self.normalizer is not None:
            #     motion = self.normalizer(motion)

            # if dataset_name == 'finedance_data':
            #     motion = (motion - self.finedance_mean) / (self.finedance_std + self.eps)
            # elif dataset_name == 'aistpp_data':
            #     motion = (motion - self.aistpp_mean) / (self.aistpp_std + self.eps)
            # elif dataset_name == 'souldance_data':
            #     motion = (motion - self.souldance_mean) / (self.souldance_std + self.eps)

            self.motions[path] = motion

        motion = self.motions[path]
        x_dict = {"x": motion, "length": len(motion)}
        return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "All_Mean.npy")
        self.std_path = os.path.join(base_dir, "All_Std.npy")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.from_numpy(np.load(self.mean_path))
        self.std = torch.from_numpy(np.load(self.std_path))

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
