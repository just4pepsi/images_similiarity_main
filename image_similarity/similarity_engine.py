__all__ = ["train_step", "test_step", "create_embedding"]

import torch


# 定义一个轮次的训练步骤
def train_step(encoder, decoder, train_loader, loss, optimizer, device):
    """
    执行一个轮次（epoch）的完整训练步骤
    :param encoder: 模型，编码器
    :param decoder: 模型，解码器
    :param train_loader: 训练数据加载器
    :param loss: 损失函数
    :param optimizer: 优化器
    :param device: 设备

    :return: 当前轮次的平均训练损失
    """
    # 设置为训练模式
    encoder.train()
    decoder.train()
    # 累计损失值
    total_loss = 0.0
    # 遍历DataLoader，按批次训练模型
    for train_imgs, target_imgs in train_loader:
        # 0. 将数据移动到设备
        train_imgs = train_imgs.to(device)
        target_imgs = target_imgs.to(device)
        # 1. 前向传播
        en_output = encoder(train_imgs)
        outputs = decoder(en_output)
        # 2. 计算损失
        loss_value = loss(outputs, target_imgs)
        # 3. 反向传播
        loss_value.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 清零梯度
        optimizer.zero_grad()
        # 6. 累加损失值
        total_loss += loss_value.item()
    return total_loss / len(train_loader)


def test_step(encoder, decoder, test_loader, loss, device):
    # 设置验证模式
    encoder.eval()
    decoder.eval()

    # 定义总测试误差
    total_loss = 0.0

    with torch.no_grad():
        for test_imgs, target_imgs in test_loader:
            test_imgs = test_imgs.to(device)
            target_imgs = target_imgs.to(device)
            # 前向传播
            en_output = encoder(test_imgs)
            outputs = decoder(en_output)
            # 计算损失
            loss_value = loss(outputs, target_imgs)
            # 累加损失
            total_loss += loss_value.item()
    return total_loss / len(test_loader)


def create_embedding(encoder, full_loader, device):
    """
    为整个数据集生成嵌入表示
    :param encoder: 训练好的编码器
    :param full_loader: 完整数据集的加载器
    :param device: 设备
    :return: 返回嵌入张量，形状（N, C, H, W）- (N, 256, 2, 2)
    """
    encoder.eval()
    # 定义嵌入张量，初始为空
    embeddings = torch.empty(0)
    with torch.no_grad():
        for train_img, target_img in full_loader:
            train_img = train_img.to(device)
            # 前向传播：编码
            encoded_img = encoder(train_img).cpu()
            # 将这一批次的特征结果拼接到嵌入张量中
            embeddings = torch.cat((embeddings, encoded_img), dim=0)
    return embeddings
