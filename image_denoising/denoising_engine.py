__all__ = ['test_step', 'train_step']

import torch


def train_step(denoiser, train_loader, loss, optimizer, device):
    """
    执行一个轮次（epoch）的完整训练步骤
    :param denoiser: 模型，降噪器
    :param train_loader: 训练数据加载器
    :param loss: 损失函数
    :param optimizer: 优化器
    :param device: 设备

    :return: 当前轮次的平均训练损失
    """
    # 设置为训练模式
    denoiser.train()
    total_loss = 0.
    for train_imgs, target_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        target_imgs = target_imgs.to(device)
        # 1. 前向传播
        outputs = denoiser(train_imgs)
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


def test_step(denoiser, test_loader, loss, device):
    denoiser.eval()
    total_loss = 0.
    with torch.no_grad():
        for test_imgs, target_imgs in test_loader:
            test_imgs = test_imgs.to(device)
            target_imgs = target_imgs.to(device)
            outputs = denoiser(test_imgs)
            loss_value = loss(outputs, target_imgs)
            total_loss += loss_value.item()
    return total_loss / len(test_loader)
