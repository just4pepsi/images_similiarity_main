import numpy as np
import torch
import torchvision.transforms as T  # 图像转换
from torch import nn, optim
from torch.utils.data import DataLoader, random_split  # 数据集和数据加载器
from tqdm import tqdm  # 进度条工具

# 导入自定义组件
from common import utils
from similarity_config import *
from similarity_data import ImageDataset
from similarity_engine import train_step, test_step, create_embedding
from similarity_model import ConvEncoder, ConvDecoder

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
    dataset = ImageDataset(IMG_PATH, transform)
    # 划分训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])
    print("--- 创建数据集完成 ---")

    # 2. 创建数据加载器
    print("--- 2. 创建数据加载器 ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    full_loader = DataLoader(dataset, batch_size=FULL_BATCH_SIZE)
    print("--- 创建数据加载器完成 ---")

    # 3. 训练模型
    # 定义模型、损失函数和优化器
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    loss = nn.MSELoss()  # 均方误差损失函数
    # 联合编码器和解码器的参数列表，传给优化器
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=LEARNING_RATE)

    encoder.to(device)
    decoder.to(device)
    min_test_loss = 9999  # 初始化最小测试误差为极大值

    print("--- 3. 开始训练模型 ---")

    for epoch in tqdm(range(EPOCHS)):
        # 训练一轮
        train_loss = train_step(encoder, decoder, train_loader, loss, optimizer, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}")

        # 进行测试，获取测试误差
        test_loss = test_step(encoder, decoder, test_loader, loss, device)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Test Loss: {test_loss}")

        # 判断当前测试误差是否小于历史最小值，如果小于则保存模型参数
        if test_loss < min_test_loss:
            print("测试误差减小了，保存模型 ...")
            min_test_loss = test_loss
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
        else:
            print("测试误差没有减小，不做保存！")

    print("--- 训练完成 ---")

    # 4. 生成图像的嵌入表达
    encoder_state_dict = torch.load(ENCODER_MODEL_NAME, map_location=device)
    encoder.load_state_dict(encoder_state_dict)

    embeddings = create_embedding(encoder, full_loader, device)
    # 转换成ndarray进行保存
    vec_embeddings = embeddings.detach().numpy().reshape(embeddings.shape[0], -1)
    # 保存到文件，生成向量数据库
    np.save(EMBEDDING_NAME, vec_embeddings)

    print("张量嵌入形状：", embeddings.shape)
    print("向量嵌入形状：", vec_embeddings.shape)
