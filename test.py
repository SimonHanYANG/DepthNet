'''
    test CellDataset
'''

import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

from CellDataset import CellDataset

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 将所有图像调整为同一大小 (96, 96)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
])

# 创建训练集和验证集
train_dataset = CellDataset(root_dir='cellData/train', transform=transform)
val_dataset = CellDataset(root_dir='cellData/val', transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

for i, (input, target) in enumerate(train_loader):
    print(input.shape)
    print(target.shape)



'''
    check image size
'''
# import os
# from PIL import Image

# def check_image_sizes(root_dir):
#     sizes = set()

#     for subdir, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith('.jpg'):  # 只处理 .jpg 文件
#                 img_path = os.path.join(subdir, file)
#                 with Image.open(img_path) as img:
#                     sizes.add(img.size)

#     return sizes

# # 使用方法
# sizes = check_image_sizes('/223010087/SimonWorkspace/paper2/depth/EPSANet/cellData/train/1/')
# for size in sizes:
#     print(size)