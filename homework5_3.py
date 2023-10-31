import cv2
import numpy as np
import glob

# 找棋盘格角点
# 棋盘格模板规格(内角点个数，内角点是和其他格子连着的点,如10 X 7)
w = 8
h = 6

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)



# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

# 标定所用图像（路径不能有中文）
images = glob.glob(r'D:\deeplearning\cv_homework\images2\*.bmp')

size = tuple()
for fname in images:
    img = cv2.imread(fname)

    # 修改图像尺寸，参数依次为：输出图像，尺寸，沿x轴，y轴的缩放系数，INTER_AREA在缩小图像时效果较好
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    size = gray.shape[::-1]  # 矩阵转置

    # 找到棋盘格角点
    # 棋盘图像(8位灰度或彩色图像)  棋盘尺寸  存放角点的位置
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if corners is None or corners.shape[0] != h*w:
        continue

    # 角点精确检测
    # criteria:角点精准化迭代过程的终止条件(阈值)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 执行亚像素级角点检测
    corners2 = cv2.cornerSubPix(gray, corners, (12, 12), (-1, -1), criteria)

    objpoints.append(objp)
    imgpoints.append(corners2)

    # 将角点在图像上显示
    cv2.drawChessboardCorners(img, (w, h), corners2, ret)
    cv2.imshow('findCorners', img)
    cv2.waitKey(100)

"""
标定、去畸变:
输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量
"""

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

# mtx：内参数矩阵
# dist：畸变系数
# rvecs：旋转向量 （外参数）
# tvecs ：平移向量 （外参数）
print("ret:", ret)
print("内参数矩阵:\n", mtx, '\n')
print("畸变系数:\n", dist, '\n')
print("旋转向量(外参数):\n", rvecs, '\n')
print("平移向量(外参数):\n", tvecs, '\n')

# 去畸变
img2 = cv2.imread(r'D:\deeplearning\cv_homework\images2\2023-10-14_14_37_18_325.bmp')
h, w = img2.shape[:2]

# 我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，
# 通过设定自由自由比例因子alpha。当alpha设为0的时候，
# 将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数

dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# 根据前面ROI区域裁剪图片
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.jpg', dst)

# 反投影误差
# 通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("total error: ", total_error / len(objpoints))

# 计算RMS误差
rms_error = np.sqrt(total_error / len(objpoints))
print(f"RMS误差: {rms_error}")