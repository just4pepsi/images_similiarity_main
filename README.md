说明：

所有包均从 PyPI 安装，torch / torchvision / torchaudio 默认会获取 CPU 版本。

如需 GPU 版本（CUDA 12.6），请使用文档中的命令单独安装：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

其他依赖包括图像处理（Pillow）、数值计算（NumPy）、Web 框架（Flask）、机器学习库（scikit-learn）等。

📘 项目介绍：智图寻宝 (Smart Image Treasure Hunt)

1. 项目概述
   “智图寻宝”是一个基于深度学习的智能商品识别系统，由尚硅谷研究院开发（版本 V1.0.4）。
   系统集成了三大核心功能：

🖼️ 图像去噪 – 自动去除图像中的随机噪声，提升图像质量

🏷️ 商品分类 – 识别五类商品（上衣、鞋、包、下装、手表）

🔍 相似商品检索 – 在数据库中查找与输入图像最相似的 5 个商品

该系统可应用于电商平台、零售智能化、图像搜索服务等场景，帮助用户快速处理图像、自动分类并检索相似商品。

2. 系统架构
   项目采用模块化设计，共包含四个主要模块：

模块 功能 技术实现
image_denoising 图像去噪 卷积自编码器（ConvDenoiser）
image_classification 商品分类 两层卷积 + 全连接分类器
image_similarity 相似检索 卷积自编码器 + KNN 向量检索
web 前后端交互 Flask Web 应用 + HTML/CSS/JS 前端
逻辑流程：

用户上传图像 → 前端展示原始图

调用去噪模块 → 展示噪声图及去噪结果

调用分类模块 → 返回商品类别

调用检索模块 → 返回最相似的 5 张商品图

3. 核心技术栈
   深度学习框架：PyTorch（模型定义、训练、推理）

Web 框架：Flask（提供 RESTful API 和页面服务）

图像处理：PIL（Pillow）、NumPy、torchvision.transforms

机器学习算法：K 近邻（KNN，基于余弦相似度）

向量存储：NumPy 数组（预计算特征嵌入）

前端：HTML、CSS、原生 JavaScript

4. 环境配置与快速启动
   4.1 创建虚拟环境（推荐）

```bash
conda create -n image_similarity_main python=3.12
conda activate image_similarity_main
```

4.2 安装依赖

```bash
pip install -r requirements.txt  # 基础依赖
# 如需 GPU 支持，额外执行：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

4.3 准备数据
将 dataset/ 目录（包含商品图像）复制到 common/ 目录下

将 fashion-labels.csv 复制到 common/ 目录下

将 logo/、pictures/ 目录复制到 web/ 目录下

4.4 训练模型（可选，项目已提供预训练权重）

```bash
# 训练去噪模型
python image_denoising/denoising_train.py

# 训练分类模型
python image_classification/classification_train.py

# 训练相似检索模型并生成向量库
python image_similarity/similarity_train.py
```

4.5 启动 Web 服务

```bash
python web/web_app.py
```

访问 http://127.0.0.1:9000 即可使用。

5. 效果演示（概要）
   步骤 操作 结果
   1 点击“上传图片” 显示原始图像
   2 点击“图像去噪” 展示添加噪声后的图像及去噪结果
   3 点击“商品分类” 文本框显示商品类别（如“上身衣服”）
   4 点击“相似商品” 下方展示 5 张数据库中相似的商品图
6. 项目亮点
   端到端实践：涵盖数据加载、模型设计、训练、部署、前端展示的全流程

模块解耦：每个功能独立成包，便于扩展和维护

轻量高效：模型参数量小，普通 CPU/GPU 均可流畅运行

教育意义：代码注释详细，适合学习 PyTorch 和计算机视觉项目开发

7. 作者与版本
   项目：尚硅谷研究院 作者：wule

版本：V1.0.4

最后更新：2026 年（根据文档推断）

该项目完整代码及资料可参考尚硅谷相关课程资源。