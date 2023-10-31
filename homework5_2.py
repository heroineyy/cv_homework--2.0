import cv2
import numpy as np

# Load the image

# 读取原始图像
original_image = cv2.imread('data/2023-10-10_14_42_36_904.bmp')

# 定义目标宽度
target_width = 800

# 计算目标高度，保持宽高比例不变
aspect_ratio = original_image.shape[1] / original_image.shape[0]
target_height = int(target_width / aspect_ratio)

# 使用cv2.resize()函数等比例放缩图像
resized_image = cv2.resize(original_image, (target_width, target_height))

if resized_image is None:
    print("Could not load image...")
    exit()

cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("input image", resized_image)

# Convert to grayscale
gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(
    gray_img,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=10,
    param1=110,
    param2=90,  # Adjust this value as needed
    minRadius=10,
    maxRadius=150
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    hough_circle = resized_image.copy()
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(hough_circle, center, radius, (0, 0, 255), 2)

    cv2.imwrite("houghcircle.jpg", hough_circle)
    cv2.imshow("houghcircle", hough_circle)

cv2.waitKey(0)
cv2.destroyAllWindows()
