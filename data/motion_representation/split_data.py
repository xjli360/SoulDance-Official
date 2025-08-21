import os
import random

npy_folder = '/workspace/HumanTOMATO/motion_data/aistpp/new_joint_vecs'
output_folder = '/workspace/HumanTOMATO/motion_data/aistpp'

npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

random.shuffle(npy_files)

total_files = len(npy_files)
train_split = int(0.7 * total_files)
val_split = int(0.2 * total_files)

train_files = npy_files[:train_split]
val_files = npy_files[train_split:train_split + val_split]
test_files = npy_files[train_split + val_split:]

def save_to_txt(file_list, file_name):
    with open(os.path.join(output_folder, file_name), 'w') as f:
        for file in file_list:
            f.write(file[:-4] + '\n')

save_to_txt(train_files, 'train.txt')
save_to_txt(val_files, 'val.txt')
save_to_txt(test_files, 'test.txt')

print(f"Saved {len(train_files)} files to train.txt")
print(f"Saved {len(val_files)} files to val.txt")
print(f"Saved {len(test_files)} files to test.txt")
