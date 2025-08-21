import os
from os.path import join as pjoin
import numpy as np

def delete_nan_data(data_dir, music_dir, data_623_dir):
    file_list = os.listdir(data_dir)
    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            os.remove(pjoin(data_dir, file))
            print(pjoin(data_dir, file))
            os.remove(pjoin(music_dir, file))
            print(pjoin(music_dir, file))
            os.remove(pjoin(data_623_dir, file))
            print(pjoin(data_623_dir, file))

# motion_723_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_723' # body263 + hands + face
# motion_623_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_623'
# motion_263_dir = '/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_263'


motion_322_dir = '/workspace/SoulNet/dataset/souldance/motion_npy'
motion_723_dir = '/workspace/SoulNet/dataset/souldance/motion_723' # body263 + hands + face
motion_623_dir = '/workspace/SoulNet/dataset/souldance/motion_623'
motion_263_dir = '/workspace/SoulNet/dataset/souldance/motion_263'

joints = 52
body_joints = 22

# data_263 = np.concatenate((data[:, :4+(body_joints - 1)*3], data[:, 4+(joints - 1)*3:4+(joints - 1)*3+(body_joints - 1)*6], data[:, 4 + (joints - 1)*9: 4 + (joints - 1) *9 + body_joints *3], data[:, -4:]), axis=1)

def bulid_723_data():
    for motion_f in os.listdir(motion_623_dir):
        if os.path.exists(os.path.join(motion_322_dir, motion_f)):
            body_motion = np.load(os.path.join(motion_263_dir, motion_f))  # 299,263
            data = np.load(os.path.join(motion_623_dir, motion_f)) 
            # 30 * 3 + 30 * 6 + 30 * 3 = 360   # 299,360
            
            hands_motion = np.concatenate((data[:, 4+(body_joints - 1)*3:4+(joints - 1)*3], data[:, 4+(joints - 1)*3+(body_joints - 1)*6:4+(joints - 1)*9], data[:, 4 + (joints - 1) *9 + body_joints *3: 4 + (joints - 1) *9 + joints *3]), axis=1)

            # face = np.load(os.path.join(motion_322_dir, motion_f))[:data.shape[0],209:309]  # 300,100

            face = np.zeros((data.shape[0],100))

            data_723 = np.concatenate((body_motion, hands_motion, face), axis=1)
            np.save(os.path.join(motion_723_dir, motion_f), data_723)
            print(os.path.join(motion_723_dir, motion_f))


bulid_723_data()



