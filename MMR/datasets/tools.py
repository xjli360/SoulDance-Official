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

def split_dataset():
    import random

    # 生成序号列表 '000000' 到 '101171'
    start, end = 0, 117680
    all_indices = [f"{i:06d}" for i in range(start, end + 1)]

    # 随机打乱序号列表
    random.shuffle(all_indices)

    # 按 8:1:1 比例划分
    total_count = len(all_indices)
    train_count = int(total_count * 0.8)
    val_count = int(total_count * 0.1)
    test_count = total_count - train_count - val_count

    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count:train_count + val_count]
    test_indices = all_indices[train_count + val_count:]


    with open('/mnt/bn/MMR/datasets/annotations/mmrdance/splits/all.txt', 'w') as all_file:
        for idx in all_indices:
            all_file.write(f"{idx}\n")

    # 将序号分别写入 train.txt, val.txt, test.txt
    with open('/mnt/bn/MMR/datasets/annotations/mmrdance/splits/train.txt', 'w') as train_file:
        for idx in train_indices:
            train_file.write(f"{idx}\n")

    with open('/mnt/bn/MMR/datasets/annotations/mmrdance/splits/val.txt', 'w') as val_file:
        for idx in val_indices:
            val_file.write(f"{idx}\n")

    with open('/mnt/bn/MMR/datasets/annotations/mmrdance/splits/test.txt', 'w') as test_file:
        for idx in test_indices:
            test_file.write(f"{idx}\n")

    print("数据已成功划分并保存到 train.txt, val.txt 和 test.txt 文件中。")



split_dataset()
