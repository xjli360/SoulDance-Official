import os
import pickle
import numpy as np

# input_folder = '/mnt/bn/code/EDGE/data/souldance/test/motions_sliced'
# output_folder = '/mnt/bn/HumanTOMATO/datasets/souldance_data/motion_npy'


def convert_aistpp_pkl(input_folder, output_folder, skip_folder=None):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            if skip_folder is not None:
                if not os.path.exists(os.path.join(skip_folder, filename.replace('pkl','npy'))):
                    print('skip :', filename)
                    continue
            input_file_path = os.path.join(input_folder, filename)
            print(input_file_path)
            output_file_path = os.path.join(output_folder, filename.replace('.pkl', '.npy'))

            with open(input_file_path, 'rb') as f:
                data = pickle.load(f)

            # pose_data = data.get('q')
            # pose_data = data.get('q').reshape(-1, 52 * 3)  # finedance
            pose_data = data.get('q').reshape(-1, 24 * 3)  # souldance  aistpp
            root_trans = data.get('pos')

            pose_len = pose_data.shape[0]
            root_orient = pose_data[:, :3]
            pose_body = pose_data[:, 3:3 + 63]
            pose_hand = np.zeros((pose_len, 90))
            pose_jaw = np.zeros((pose_len, 3))
            face_expr = np.zeros((pose_len, 50))
            face_shape = np.zeros((pose_len, 100))

            betas = np.zeros((pose_len, 10))
            motionx_data = np.concatenate((root_orient, pose_body, pose_hand, pose_jaw, face_expr, face_shape, root_trans, betas), axis=1)
            print(motionx_data.shape)
            if pose_data is not None:
                np.save(output_file_path, motionx_data)
                print(f'Saved {output_file_path}')
            else:
                print(f'No pose data found in {filename}')

def convert_smpl_joints(source_joints_dir, save_dir, skip_dir):
    os.makedirs(save_dir, exist_ok=True)
    for j in os.listdir(source_joints_dir):
        if not os.path.exists(os.path.join(skip_dir, j)):
            print('skip :', j)
            continue
        sj = np.load(os.path.join(source_joints_dir, j))
        sj = sj.reshape(sj.shape[0], -1, 3)
        pose_zero = np.zeros((sj.shape[0], 144 - sj.shape[1], 3))
        pose = np.concatenate((sj, pose_zero), axis=1)
        np.save( os.path.join(save_dir, j), pose)


if __name__ == "__main__":
    input_folder = '/workspace/SoulNet/dataset/edge_aistpp/test/motions_sliced'
    output_folder = '/workspace/SoulNet/dataset/edge_aistpp/motion_npy'

    convert_aistpp_pkl(input_folder, output_folder)

    # source_joints_dir = '/mnt/bn/Bailando/experiments/actor_critic/eval/pkl/ep000010_sliced/'
    # save_dir = '/mnt/bn/Bailando/experiments/actor_critic/eval/pkl/joint/'
    # skip_dir = '/mnt/bn/HumanTOMATO/datasets/aistpp_data/jukebox_feats'

    # convert_smpl_joints(source_joints_dir, save_dir, skip_dir)

