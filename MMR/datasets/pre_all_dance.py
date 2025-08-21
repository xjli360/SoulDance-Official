# Copyright (c) 2023 Mathis Petrovich

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

import os
import random
import json

def get_random_files(folder, ratio):
    """Randomly select files from folder by ratio"""
    files = os.listdir(folder)
    num_to_select = int(len(files) * ratio)
    return random.sample(files, num_to_select)

# Folder paths
folders = [
    '/mnt/bn/datasets/finedance_data/jukebox_feats',
    '/mnt/bn/datasets/aistpp_data/jukebox_feats',
    '/mnt/bn/datasets/souldance_data/jukebox_feats'
]

ratios = [1, 1, 0.6]

annotations = {}

file_counter = 0

for folder, ratio in zip(folders, ratios):
    selected_files = get_random_files(folder, ratio)
    for file_name in selected_files:
        line = os.path.join(folder, file_name)
        annotations[f"{file_counter:06d}"] = {"path": line}
        file_counter += 1

with open("/mnt/bn/MMR/datasets/annotations/mmrdance/annotations.json", "w") as json_file:
    json.dump(annotations, json_file, indent=4)

print("annotations.json file has been generated.")
