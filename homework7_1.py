import cv2
import numpy as np

def visualize_matches(image, match_locations, method, threshold):
    # 在图像上绘制匹配位置的红色框线
    for loc in match_locations:
        top_left = loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

    # 在图像上标注使用的方法和阈值
    text = f"{method} (Threshold: {threshold})"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

# 加载图像
image = cv2.imread('data/round.jpg')  # 加载原始图像
template = cv2.imread('data/template.jpg')  # 加载模板图像

# 模板匹配 - SSD
ssd_result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
min_val_ssd, max_val_ssd, min_loc_ssd, max_loc_ssd = cv2.minMaxLoc(ssd_result)

# 设置SSD阈值
ssd_thresholds = [0.6, 0.5, 0.9]  # 不同的SSD阈值

# 模板匹配 - NCC
ncc_result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
min_val_ncc, max_val_ncc, min_loc_ncc, max_loc_ncc = cv2.minMaxLoc(ncc_result)

# 设置NCC阈值
ncc_thresholds = [0.8, 0.9, 0.95]  # 不同的NCC阈值

# 创建一个空白画布，用于显示所有结果
canvas = np.zeros((image.shape[0] * 2, image.shape[1] * 3, 3), dtype=np.uint8)

# SSD匹配结果可视化
for i, threshold in enumerate(ssd_thresholds):
    ssd_match_locations = np.where(ssd_result <= threshold)
    ssd_match_locations = list(zip(*ssd_match_locations[::-1]))
    result_image = image.copy()
    result_image = visualize_matches(result_image, ssd_match_locations, 'SSD', threshold)
    row = i // 3  # 计算行索引
    col = i % 3  # 计算列索引
    canvas[row * image.shape[0]: (row + 1) * image.shape[0], col * image.shape[1]: (col + 1) * image.shape[1], :] = result_image

# NCC匹配结果可视化
for i, threshold in enumerate(ncc_thresholds):
    ncc_match_locations = np.where(ncc_result >= threshold)
    ncc_match_locations = list(zip(*ncc_match_locations[::-1]))
    result_image = image.copy()
    result_image = visualize_matches(result_image, ncc_match_locations, 'NCC', threshold)
    row = (i + 3) // 3  # 计算行索引
    col = (i + 3) % 3  # 计算列索引
    canvas[row * image.shape[0]: (row + 1) * image.shape[0], col * image.shape[1]: (col + 1) * image.shape[1], :] = result_image

# 显示结果图像
cv2.imshow('Template Matching Results', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()