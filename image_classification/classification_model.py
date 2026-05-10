__all__ = ["Classifier"]

import torch
import torch.nn as nn


# 定义模型 Sequential 实现
class Classifier(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, n_classes)
        )

    def forward(self, x):
        return self.model(x)


# 测试模型
if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)  # 输入数据
    model = Classifier()  # 模型
    output = model(x)  # 前向传播
    print(output.size())
