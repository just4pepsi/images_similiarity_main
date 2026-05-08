import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T  # 图像转换
from torch.utils.data import DataLoader, random_split  # 数据集和数据加载器

# 导入自定义组件
from common import utils
from denoising_config import *
from denoising_data import *
from denoising_model import *


# 定义函数：用一批数据进行测试，并画图
def test(denoiser, test_loader, device):
    denoiser.eval()
    # 1. 从DataLoader中取出一个批次的数据
    data_iter = iter(test_loader)
    noise_images, original_images = next(data_iter)
    print("测试数据形状：", noise_images.shape)

    denoiser = denoiser.to(device)
    noise_images = noise_images.to(device)
    # 2. 前向传播，进行推理重建图像
    outputs = denoiser(noise_images)
    print("重构图像形状：", outputs.shape)

    # 3. 画图
    # 3.1 噪声图像
    noise_imgs = noise_images.permute(0, 2, 3, 1).cpu().numpy()
    print("转换后的噪声图像形状：", noise_imgs.shape)
    # 3.2 去噪图像
    output_imgs = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    print("转换后的去噪图像形状：", output_imgs.shape)
    # 3.3 原始图像
    original_imgs = original_images.permute(0, 2, 3, 1).cpu().numpy()
    print("转换后的原始图像形状：", original_imgs.shape)
    fig, axes = plt.subplots(3, 10, figsize=(25, 4))
    for imgs, row in zip([noise_imgs, output_imgs, original_imgs], axes):
        for img, ax in zip(imgs, row):
            ax.imshow(img)
            ax.axis('off')
    plt.show()


if __name__ == '__main__':
    # 0. 准备工作
    # 检测GPU是否可用并定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 指定随机数种子，去除训练的不确定性
    utils.seed_everything(SEED)

    # 定义图像预处理操作
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])

    # 1. 创建数据集
    print("--- 1. 创建数据集 ---")
    dataset = ImageDataSet(IMG_PATH, transform)
    # 划分训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])
    print("--- 创建数据集完成 ---")

    # 2. 创建数据加载器
    print("--- 2. 创建数据加载器 ---")
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("--- 创建数据加载器完成 ---")

    # 3. 加载模型
    loaded_denoiser = ConvDenoiser()
    print("--- 3. 从文件加载模型 ---")
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    print("--- 加载模型完成 ---")

    loaded_denoiser.to(device)
    # 4. 测试
    print("--- 4. 测试结果如下 ---")
    test(loaded_denoiser, test_loader, device)
