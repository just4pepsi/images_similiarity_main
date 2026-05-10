# 数据预处理配置
FASHION_LABELS_PATH = "../common/fashion-labels.csv"
IMG_PATH = "../common/dataset/"
IMG_HEIGHT = 64
IMG_WIDTH = 64

# 随机性与数据集划分
SEED = 42  # 随机数种子
TRAIN_RATIO = 0.75  # 训练集划分比例
TEST_RATIO = 1 - TRAIN_RATIO

# 训练超参数设置
LEARNING_RATE = 0.001  # 学习率
EPOCHS = 20  # 训练总轮次
TRAIN_BATCH_SIZE = 128  # mini-batch大小
TEST_BATCH_SIZE = 128

# 模块名称和保存模型参数的文件名
PACKAGE_NAME = 'image_classification'
CLASSIFIER_MODEL_NAME = 'classifier.pt'

# 数字标签对应分类名称的字典
classification_names = {
    0: '上身衣服',  # 数字 0 对应“上身衣服”
    1: '鞋',  # 数字 1 对应“鞋”
    2: '包',  # 数字 2 对应“包”
    3: '下身衣服',  # 数字 3 对应“下身衣服”
    4: '手表'  # 数字 4 对应“手表”
}
