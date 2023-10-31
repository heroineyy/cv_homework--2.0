import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取两张图像
image1 = cv2.imread('data/1.jpg')
image2 = cv2.imread('data/2.jpg')

# 使用ORB等方法提取特征点和匹配
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 提取特征点的坐标
points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# 设置不同的RANSAC参数
ransac_parameters = [
    (3.0, 1000, 0.99),
    (5.0, 1000, 0.99),
    (3.0, 2000, 0.99)
]

# 创建一个 Matplotlib 图形窗口
plt.figure(figsize=(15, 10))

for i, (threshold, max_trials, confidence) in enumerate(ransac_parameters):
    # 使用RANSAC估计基础矩阵
    fundamental_matrix, inliers = cv2.findFundamentalMat(
        points1, points2, cv2.FM_RANSAC, threshold, confidence, max_trials)

    # 从匹配中筛选出内点
    points1_inliers = points1[inliers.ravel() == 1]
    points2_inliers = points2[inliers.ravel() == 1]

    # 绘制匹配和内点
    matches_inliers = [matches[j] for j in range(len(matches)) if inliers[j] == 1]
    image_with_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, outImg=None)
    image_with_inliers = cv2.drawMatches(image1, kp1, image2, kp2, matches_inliers, outImg=None)

    # 创建子图
    plt.subplot(3, 2, 2*i+1)
    plt.imshow(image_with_matches)
    plt.title('All Matches (Threshold={})'.format(threshold), color='red')
    plt.subplot(3, 2, 2*i+2)
    plt.imshow(image_with_inliers)
    plt.title('Inliers (Threshold={})'.format(threshold), color='red')

    # 打印基础矩阵和内点数
    print("RANSAC Parameters: Threshold={}, Max Trials={}, Confidence={}".format(
        threshold, max_trials, confidence))
    print("Estimated Fundamental Matrix:")
    print(fundamental_matrix)
    print("Number of Inliers:", len(points1_inliers))

# 显示 Matplotlib 图形窗口
plt.show()
