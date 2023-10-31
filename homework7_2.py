import gzip
import urllib

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# 1. 加载MNIST数据集
def load_mnist():
    train_images_file = "data/train-images-idx3-ubyte.gz"
    train_labels_file = "data/train-labels-idx1-ubyte.gz"
    test_images_file = "data/t10k-images-idx3-ubyte.gz"
    test_labels_file = "data/t10k-labels-idx1-ubyte.gz"

    with gzip.open(train_images_file, 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(train_labels_file, 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(test_images_file, 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(test_labels_file, 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_images, train_labels, test_images, test_labels


# 2. 加载MNIST数据集
train_images, train_labels, test_images, test_labels = load_mnist()

# 3. 提取HOG描述子特征
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(hog_feature)
    return np.array(hog_features)


train_features = extract_hog_features(train_images)
test_features = extract_hog_features(test_images)

# 4. 使用KNN算法进行聚类
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_features, train_labels)
knn_predictions = knn_classifier.predict(test_features)
knn_accuracy = accuracy_score(test_labels, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

# 5. 使用SVM算法进行分类
svm_classifier = SVC()
svm_classifier.fit(train_features, train_labels)
svm_predictions = svm_classifier.predict(test_features)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print("SVM Accuracy:", svm_accuracy)