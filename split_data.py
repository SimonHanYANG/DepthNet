'''
If using your own dataset, make sure your data is stored in the folder which name is the class name
then, using this file to split into train/ and val/ dataset
'''

import os
import shutil
import random

def split_dataset(root_dir, train_dir='cellData/train/', val_dir='cellData/val/', split_ratio=0.8):
    # 确保输出目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)

        # 检查是否是文件夹
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

        # 随机打乱图片
        random.shuffle(images)

        # 计算训练集的大小
        train_size = int(len(images) * split_ratio)

        # 分割图片
        train_images = images[:train_size]
        val_images = images[train_size:]

        # 复制图片到对应的输出目录
        for image in train_images:
            src_path = os.path.join(class_path, image)
            dst_path = os.path.join(train_dir, class_dir, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        for image in val_images:
            src_path = os.path.join(class_path, image)
            dst_path = os.path.join(val_dir, class_dir, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)


# 使用方法
split_dataset('./sperm-depth/')