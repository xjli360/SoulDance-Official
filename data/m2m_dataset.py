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

import pdb
from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from .rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
import random
import codecs as cs


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class DanceDataset(data.Dataset):
    def __init__(self, opt, split_file,mean=None, std=None):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # motion = np.array(pickle.load(pjoin(opt.motion_dir, name + '.pkl').get('smpl_poses')))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def motion_merge(self, body_hands):
        # 30 * 3 + 30 * 6 + 30 * 3 = 360
        # hands_motion = np.concatenate((data[:, 4+(body_joints - 1)*3:4+(joints - 1)*3], data[:, 4+(joints - 1)*3+(body_joints - 1)*6:4+(joints - 1)*9], data[:, 4 + (joints - 1) *9 + body_joints *3: 4 + (joints - 1) *9 + joints *3]), axis=1)
        # 4 + 21 *3 + 21 * 6 + 22*3 + 4 =  263
        # data_263 = np.concatenate((data[:, :4+(body_joints - 1)*3], data[:, 4+(joints - 1)*3:4+(joints - 1)*3+(body_joints - 1)*6], data[:, 4 + (joints - 1)*9: 4 + (joints - 1) *9 + body_joints *3], data[:, -4:]), axis=1)
        body = body_hands[..., :263]
        hands = body_hands[..., 263:623]

        # motion_623 = np.concatenate((body[:, :4+(22 - 1)*3], 
        #                             hands[:, :30*3], 
        #                             body[:, 4+(22 - 1)*3:4+(22 - 1)*3+(22 - 1)*6], 
        #                             hands[:, 30*3:30*6],
        #                             body[:, 4+(22 - 1)*9:4+(22 - 1)*9+22*3], 
        #                             hands[:, 30*9:30*9+30*3], 
        #                             body[:, -4:]), axis=1)

        motion_623 = torch.cat((body[..., :4+(22 - 1)*3], 
                                hands[..., :30*3], 
                                body[..., 4+(22 - 1)*3:4+(22 - 1)*3+(22 - 1)*6], 
                                hands[..., 30*3:30*9],
                                body[..., 4+(22 - 1)*9:4+(22 - 1)*9+22*3], 
                                hands[..., 30*9:30*9+30*3], 
                                body[..., -4:]), dim=-1)
        return motion_623


    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        # motion = (motion - self.mean) / self.std

        return motion



def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class SoulDanceDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        joints = 52
        body_joints = 22
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # body + face
                # data_263 = np.concatenate((data[:, :4+(body_joints - 1)*3], data[:, 4+(joints - 1)*3:4+(joints - 1)*3+(body_joints - 1)*6], data[:, 4 + (joints - 1)*9: 4 + (joints - 1) *9 + body_joints *3], data[:, -4:]), axis=1)
                # face = np.load(pjoin(opt.face_dir, name + '.npy'))[:motion.shape[0],209:309]
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            print('feat_bias:', opt.feat_bias)
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean
    

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion



class Music2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) > 300):
                    continue
                # music_feat = np.load(pjoin(opt.music_dir, name + '.npy'))  # jukebox
                music_feat = np.load(pjoin(opt.music_dir, name + '.npy'))
                # music_feat = np.load(pjoin(opt.music_dir, name + '.npy')).mean(axis=-2)  # baseline 
                # TODO add music_feat

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'music_feat': music_feat,
                                   'music_name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):

        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, music_feat, music_name = data['motion'], data['length'], data['music_feat'], data['music_name']


        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        if self.opt.dataset_name == "aistpp":
            motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return music_feat,music_name, motion, m_length



class Music2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                # import pdb;pdb.set_trace()
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) > 300):
                    continue
                music_feat = np.load(pjoin(opt.music_dir, name + '.npy'))
                # music_feat = np.load(pjoin(opt.music_dir, name + '.npy')).mean(axis=-2)
                # TODO add music_feat
                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'music_feat': music_feat}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass
        
        # import pdb;pdb.set_trace()
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, music_feat = data['motion'], data['length'], data['music_feat']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        if self.opt.dataset_name == "aistpp":
            motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return music_feat, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)