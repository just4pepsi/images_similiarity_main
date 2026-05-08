# 定义模块的公开接口，仅暴露 ImageDataSet 类
__all__ = ["ImageDataSet"]

import os  # 导入os库，用于处理文件和目录路径
import re

import torch
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
from torch.utils.data import Dataset  # 从PyTorch的工具库中导入Dataset类，用于自定义数据集

import denoising_config


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name: [convert(c) for c in re.split('([0-9]+)', img_name)]
    return sorted(data, key=alphanum_key)


class ImageDataSet(Dataset):
    def __init__(self, img_dir, transform=None):
        self.main_dir = img_dir
        self.transform = transform
        self.images_names = sorted_alphanumeric(os.listdir(img_dir))

    def __len__(self):
        return len(self.images_names)

    # 定义__getitem__方法，根据索引idx获取图像
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.images_names[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            raise ValueError("transform参数不能为None，需指定预处理方法")
        ## 向输入图像添加随机噪声
        # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
        noisy_imgs = tensor_image + denoising_config.NOISE_FACTOR * torch.randn_like(tensor_image)
        # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
        noisy_imgs = torch.clip(noisy_imgs, 0., 1.)

        return noisy_imgs, tensor_image
