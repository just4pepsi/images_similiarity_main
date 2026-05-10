# 数据预处理配置
IMG_PATH = "../common/dataset/"
IMG_HEIGHT = 64
IMG_WIDTH = 64

# 随机性与数据集划分
SEED = 42  # 随机数种子
TRAIN_RATIO = 0.75  # 训练集划分比例
TEST_RATIO = 1 - TRAIN_RATIO

# 训练超参数设置
LEARNING_RATE = 0.001  # 学习率
EPOCHS = 30  # 训练总轮次
TRAIN_BATCH_SIZE = 32  # mini-batch大小
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

# 模块名称和保存模型参数的文件名
PACKAGE_NAME = 'image_similarity'
ENCODER_MODEL_NAME = 'deep_encoder.pt'
DECODER_MODEL_NAME = 'deep_decoder.pt'
EMBEDDING_NAME = 'data_embedding.npy'
