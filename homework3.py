import cv2
import numpy as np
from scipy.signal import convolve
from scipy.signal import find_peaks


def smooth_histogram(histogram, kernel_size=5):
    # 创建高斯滤波器内核
    kernel = np.exp(-np.linspace(-1, 1, kernel_size)**2 / 2)
    kernel /= kernel.sum()
    # 对直方图应用滤波器
    smoothed_histogram = convolve(histogram, kernel, mode='same')
    return smoothed_histogram

def segment_image_with_threshold(image):
    # 将输入图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算图像的灰度直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    # 寻找直方图中的峰值
    smoothed_histogram = smooth_histogram(hist)
    #smoothed_image = cv2.medianBlur(gray, 15)
    # 使用SciPy的find_peaks函数找到平滑后的直方图中的峰值
    peaks, _ = find_peaks(smoothed_histogram, distance=30)  # 调整distance以适应图像
    # 如果找到了两个峰值，则将它们作为双阈值返回
    if len(peaks) >= 2:
        lower_threshold = int(peaks[0])
        upper_threshold = int(peaks[1])
        # 使用双阈值法进行分割
        _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        # 腐蚀操作
        eroded_image = cv2.erode(binary, kernel, iterations=2)
        # 膨胀操作
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=5)
        result_image = cv2.bitwise_and(image, image, mask=dilated_image)
        return result_image
    else:
        # 找不到两个峰值，返回None
        print("无法找到足够的峰值来确定阈值。")
        return None

def region_growing_segmentation(image):
    seed_point = (240, 240)  # 种子点的坐标 (x, y)
    threshold = 5  # 生长条件的阈值
    # 将输入图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建标记图像，初始化为0（未标记）
    height, width = gray.shape
    segmented_image = np.zeros((height, width), dtype=np.uint8)
    # 创建队列并将种子点入队
    queue = []
    queue.append(seed_point)
    while len(queue) > 0:
        # 取出队列中的一个像素点
        current_point = queue.pop(0)
        # 获取当前像素的灰度值
        current_value = gray[current_point[1], current_point[0]]
        # 检查相邻像素，根据生长条件合并
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = current_point[0] + dx
                y = current_point[1] + dy
                # 确保像素在图像范围内
                if x >= 0 and y >= 0 and x < width and y < height:
                    if segmented_image[y, x] == 0 and abs(int(gray[y, x]) - int(current_value)) < threshold:
                        # 符合条件，标记像素并入队
                        segmented_image[y, x] = 255
                        queue.append((x, y))
    result_image = cv2.bitwise_and(image, image, mask=segmented_image)
    return result_image

def otsu_segmentation(image):
    # 将输入图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用Otsu算法计算阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 定义腐蚀和膨胀的核（kernel）
    kernel = np.ones((5, 5), np.uint8)
    # 腐蚀操作
    eroded_image = cv2.erode(binary, kernel, iterations=5)
    # 膨胀操作
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=5)
    result_image = cv2.bitwise_and(image, image, mask=dilated_image)
    return result_image

def watershed_segmentation(image):
    # 将输入图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用OTSU阈值法进行二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 形态学操作去除噪声和连接区域
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 确定背景和前景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    # 确定未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 标记前景对象
    _, markers = cv2.connectedComponents(sure_fg)
    # 将标记加1以避免与背景冲突
    markers += 1
    # 将未知区域标记为0
    markers[unknown == 255] = 0
    # 使用分水岭算法进行分割
    markers = cv2.watershed(image, markers)
    # 创建一个新的图像来保存前景区域
    segmented_image = np.zeros_like(image)
    segmented_image[markers == 1] = image[markers == 1]  # 将前景区域复制到新图像
    return segmented_image


if __name__ == "__main__":
    # 选择分割方式
    segment_way = 1
    if segment_way == 1:
        # 题目一：分割狒狒鼻子
        input_image = cv2.imread('data/baboon.png')
        segmented_image = segment_image_with_threshold(input_image)
    if segment_way == 2:
        # 题目二： 区域生长法
        input_image = cv2.imread('data/balloon.bmp')
        segmented_image = region_growing_segmentation(input_image)
    if segment_way == 3:
        # 题目三： Otsu法
        input_image = cv2.imread('data/balloon.bmp')
        segmented_image = otsu_segmentation(input_image)
    if segment_way == 4:
        # 题目四 ：分水邻算法
        input_image = cv2.imread('data/orange.bmp')
        segmented_image = watershed_segmentation(input_image)

    if segmented_image is not None:
        # 显示分割结果
        cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


























