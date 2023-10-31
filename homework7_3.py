import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import numpy as np
import pickle

# 下载并加载CIFAR-10数据集
def load_cifar10():

    # 解压缩数据集
    import tarfile
    with tarfile.open("data/cifar-10-python.tar.gz", "r:gz") as tar:
        tar.extractall()

    # 加载训练集
    train_images = None
    train_labels = []
    for batch_id in range(1, 6):
        with open(f"cifar-10-batches-py/data_batch_{batch_id}", "rb") as file:
            batch = pickle.load(file, encoding="bytes")
        if train_images is None:
            train_images = batch[b"data"]
        else:
            train_images = np.vstack((train_images, batch[b"data"]))
        train_labels += batch[b"labels"]

    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 调整图像数组的形状和通道顺序
    train_labels = np.array(train_labels)

    # 加载测试集
    with open("cifar-10-batches-py/test_batch", "rb") as file:
        test = pickle.load(file, encoding="bytes")
    test_images = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 调整图像数组的形状和通道顺序
    test_labels = np.array(test[b"labels"])

    return train_images, train_labels, test_images, test_labels

# 提取HOG特征
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
        hog_features.append(hog_feature)
    return np.array(hog_features)

# 加载CIFAR-10数据集
train_images, train_labels, test_images, test_labels = load_cifar10()

# 提取训练集和测试集的HOG特征
train_features = extract_hog_features(train_images)
test_features = extract_hog_features(test_images)

# 创建SVM分类器
svm_classifier = SVC()

# 训练SVM模型（使用HOG特征）
print("正在训练（使用HOG特征）")
svm_classifier.fit(train_features, train_labels)
print("训练结束（使用HOG特征）")

# 在测试集上进行预测（使用HOG特征）
print("正在预测（使用HOG特征）")
predictions_hog = svm_classifier.predict(test_features)

# 计算使用HOG特征的准确率
accuracy_hog = accuracy_score(test_labels, predictions_hog)
print("Accuracy（使用HOG特征）:", accuracy_hog)

# 创建另一个SVM分类器
svm_classifier_no_hog = SVC()

# 将图像转换为一维向量表示
train_images_flat = train_images.reshape(len(train_images), -1)
test_images_flat = test_images.reshape(len(test_images), -1)

# 训练SVM模型（不使用HOG特征）
print("正在训练（不使用HOG特征）")
svm_classifier_no_hog.fit(train_images_flat, train_labels)
print("训练结束（不使用HOG特征）")

# 在测试集上进行预测（不使用HOG特征）
print("正在预测（不使用HOG特征）")
predictions_no_hog = svm_classifier_no_hog.predict(test_images_flat)

# 计算不使用HOG特征的准确率
accuracy_no_hog = accuracy_score(test_labels, predictions_no_hog)
print("Accuracy（不使用HOG特征）:", accuracy_no_hog)

# 比较使用HOG特征和不使用HOG特征的准确率差异
accuracy_difference = accuracy_hog - accuracy_no_hog
print("Accuracy差异:", accuracy_difference)