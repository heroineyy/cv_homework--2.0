import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('data/2023-09-19_14_51_16_763.bmp')

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测
edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 将直线特征点提取出来
line_points = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    line_points.append([x1, y1])
    line_points.append([x2, y2])

line_points = np.array(line_points)

# 自定义RANSAC参数
n_iterations = 1000  # 迭代次数
sample_size = 2  # 随机样本的大小
threshold = 1.0  # 内点阈值

best_model = None
best_inliers = []

for _ in range(n_iterations):
    # 随机选择两个点作为候选直线的参数
    sample_indices = np.random.choice(len(line_points), sample_size, replace=False)
    sample = line_points[sample_indices]

    # 拟合直线模型
    x1, y1 = sample[0]
    x2, y2 = sample[1]

    # 计算直线的斜率和截距
    if x2 - x1 == 0:
        continue  # 避免除以零的情况
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # 计算其他点到直线的距离
    distances = np.abs(line_points[:, 1] - (m * line_points[:, 0] + b))

    # 确定内点
    inlier_indices = np.where(distances < threshold)[0]

    # 更新最佳模型
    if len(inlier_indices) > len(best_inliers):
        best_model = (m, b)
        best_inliers = inlier_indices

# 最佳模型的参数
best_m, best_b = best_model

# 输出拟合的直线参数
print("Best Fit Line: y =", best_m, "x +", best_b)

# 绘制拟合直线和内点
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(line_points[:, 0], line_points[:, 1], label="Data")
plt.scatter(line_points[best_inliers, 0], line_points[best_inliers, 1], color='r', label="Inliers")
x_values = np.array([min(line_points[:, 0]), max(line_points[:, 0])])
y_values = best_m * x_values + best_b
plt.plot(x_values, y_values, color='g', label="RANSAC Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
