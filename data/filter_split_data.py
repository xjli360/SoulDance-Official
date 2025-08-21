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

import glob
import os
import pickle
import shutil
from pathlib import Path


def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out


filter_list = set(fileToList("splits/ignore_list.txt"))
train_list = set(fileToList("splits/crossmodal_train.txt"))
test_list = set(fileToList("splits/crossmodal_test.txt"))


def split_data(dataset_path):
    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            if sequence in filter_list:
                continue
            motion = f"{dataset_path}/motions/{sequence}.pkl"
            wav = f"{dataset_path}/wavs/{sequence}.wav"
            assert os.path.isfile(motion)
            assert os.path.isfile(wav)
            motion_data = pickle.load(open(motion, "rb"))
            trans = motion_data["smpl_trans"]
            pose = motion_data["smpl_poses"]
            scale = motion_data["smpl_scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"{split_name}/wavs/{sequence}.wav")
