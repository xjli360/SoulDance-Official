from data.m2m_dataset import Music2MotionDatasetEval, collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_music_motion_loader(opt_path, batch_size, mean, std, fname, device):
    opt = get_opt(opt_path, device)

    if opt.dataset_name == 'souldance' or opt.dataset_name == 'finedance' or opt.dataset_name == 'aistpp':
        print('Loading dataset %s ...' % opt.dataset_name)
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Music2MotionDatasetEval(opt, mean, std, split_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,collate_fn=collate_fn, shuffle=True, pin_memory=True)
    else:
        print(opt.dataset_name)
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset