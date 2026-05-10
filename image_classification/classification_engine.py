__all__ = ["train_step", "test_step"]

import torch


def train_step(classifier, train_loader, loss, optimizer, device):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)  # 前向传播
        loss_value = loss(outputs, labels)  # 计算损失
        loss_value.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 梯度清零
        total_loss += loss_value.item()
    return total_loss / len(train_loader)


def test_step(classifier, test_loader, loss, device):
    total_loss = 0
    correct_num = 0  # 累计分类准确的数量

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)  # 前向传播
            loss_value = loss(outputs, labels)  # 计算损失
            total_loss += loss_value.item()
            # 预测分类标签，并计算正确个数
            pred = outputs.argmax(dim=1)
            correct_num += pred.eq(labels).sum()

    return total_loss / len(test_loader), correct_num
