import os
from argparse import Namespace
import re
from os.path import join as pjoin
from utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 'souldance': # body only
        opt.data_root = './dataset/souldance/'
        opt.motion_dir = pjoin(opt.data_root, 'motion_723')
        opt.music_dir = '/mnt/bn/MMR/datasets/mmr_souldance_music_feats'
        opt.joints_num = 52
        opt.dim_pose = 723
        opt.max_motion_length = 300
        opt.max_motion_frame = 300
        opt.max_motion_token = 55
    elif opt.dataset_name == 'finedance':
        opt.data_root = './dataset/finedance/'
        opt.motion_dir = pjoin(opt.data_root, 'motion_263')
        opt.music_dir = '/mnt/bn/MMR/datasets/mmr_finedance_feats'
        opt.joints_num = 52
        opt.dim_pose = 623
        opt.max_motion_length = 150
        opt.max_motion_frame = 150
        opt.max_motion_token = 55

    elif opt.dataset_name == "aistpp":
        opt.data_root = './dataset/aistpp/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.music_dir = '/mnt/bn/MMR/datasets/mmr_music_feats'
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 300
        opt.max_motion_frame = 300
        opt.max_motion_token = 55
    else:
        raise KeyError('Dataset not recognized')
    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs) # Overwrite with kwargs params

    return opt