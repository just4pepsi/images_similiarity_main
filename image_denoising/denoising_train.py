import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T  # 图像转换
from torch.utils.data import DataLoader, random_split  # 数据集和数据加载器
from tqdm import tqdm  # 进度条工具

# 导入自定义组件
from common import utils
from denoising_config import *
from denoising_data import *
from denoising_engine import train_step, test_step
from denoising_model import *
from image_denoising.denoising_test import test

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.seed_everything(SEED)

    # 定义图像预处理流程
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
    ])
    print("------------ 数据集开始创建 ------------")
    dataset = ImageDataSet(IMG_PATH, transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])
    print("------------ 数据加载器开始创建 ------------")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("------------ 模型开始创建 ------------")
    denoiser = ConvDenoiser()
    loss = nn.MSELoss()
    optimizer = optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)
    denoiser.to(device)
    # 初始化最佳损失值为极大值
    min_loss = 9999
    print("------------ 训练开始 ------------")
    for epoch in tqdm(range(EPOCHS)):
        # 训练一轮
        train_loss = train_step(denoiser, train_loader, loss, optimizer, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}")
        # 进行测试，获取测试误差
        test_loss = test_step(denoiser, test_loader, loss, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Test Loss: {test_loss}")
        # 判断当前测试误差是否小于历史最小值，如果小于则保存模型参数
        if test_loss < min_loss:
            print("------------ 模型保存 ------------")
            min_loss = test_loss
            torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)

    print("--- 训练完成 ---")

    # 4. 增加测试效果展示
    print("--- 4. 测试模型效果 ---")
    test(denoiser, test_loader, device)

    # 5. 对比：从文件加载模型测试效果展示
    loaded_denoiser = ConvDenoiser()
    print("--- 5. 从文件加载模型 ---")
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    print("--- 加载模型完成 ---")

    loaded_denoiser.to(device)
    print("--- 测试结果如下 ---")
    test(loaded_denoiser, test_loader, device)
