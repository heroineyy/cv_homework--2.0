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

# 提取直线特征点
line_points = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    line_points.append([x1, y1])
    line_points.append([x2, y2])

line_points = np.array(line_points)
# change_start
# 添加一些外点（模拟异常值）
num_outliers = 20
outlier_indices = np.random.choice(len(line_points), num_outliers, replace=False)
line_points[outlier_indices] = np.random.uniform(0, 800, (num_outliers, 2))


# Robust最小二乘法拟合
def robust_linear_regression(x, y, delta, num_iterations):
    m = len(x)
    theta = np.zeros(2)  # 初始参数

    for _ in range(num_iterations):
        y_pred = theta[0] + theta[1] * x
        residuals = y - y_pred
        absolute_residual = np.abs(residuals)
        weights = np.where(absolute_residual <= delta, 1.0, delta / absolute_residual)

        # 使用加权的最小二乘法拟合
        X = np.vstack((np.ones(m), x)).T
        W = np.diag(weights)
        A = np.dot(X.T, np.dot(W, X))
        b = np.dot(X.T, np.dot(W, y))

        # 计算参数更新
        delta_theta = np.linalg.solve(A, b)
        theta += delta_theta

    return theta


# 使用Robust最小二乘法拟合
delta = 1  # Huber损失函数的参数
num_iterations = 100
best_theta = robust_linear_regression(line_points[:, 0], line_points[:, 1], delta, num_iterations)/num_iterations

# 绘制拟合结果
# 绘制拟合结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(line_points[:, 0], line_points[:, 1], label="Data")
x_values = np.array([min(line_points[:, 0]), max(line_points[:, 0])])
y_values = best_theta[0] + best_theta[1] * x_values
plt.plot(x_values, y_values, color='r', label="Robust Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

