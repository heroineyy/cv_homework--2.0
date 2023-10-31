import numpy as np
import random
import os
from keras.models import load_model
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score, jaccard_score

def batch_data_test(input_name, n, batch_size=8, input_size_1=256, input_size_2=256):
    rand_num = random.randint(0, n - 1)
    img1 = io.imread('D:/deeplearning/datasets/unet/imgs2/test/' + input_name[rand_num]).astype("float")
    img2 = io.imread('D:/deeplearning/datasets/unet/masks2/test/' + input_name[rand_num]).astype("float")
    img1 = resize(img1, [input_size_1, input_size_2, 3])
    img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0, n - 1)
        img1 = io.imread('D:/deeplearning/datasets/unet/imgs2/test/' + input_name[rand_num]).astype("float")
        img2 = io.imread('D:/deeplearning/datasets/unet/masks2/test/' + input_name[rand_num]).astype("float")
        img1 = resize(img1, [input_size_1, input_size_2, 3])
        img2 = resize(img2, [input_size_1, input_size_2, 3])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis=0)
        batch_output = np.concatenate((batch_output, img2), axis=0)
    return batch_input, batch_output


# 加载已训练的模型
model = load_model('unet.h5')

test_name = os.listdir('D:/deeplearning/datasets/unet/imgs2/test/')
n_test = len(test_name)

test_X, test_Y = batch_data_test(test_name, n_test, batch_size=1)

# 进行预测
pred_Y = model.predict(test_X)

# 计算均方误差
mse = np.mean((test_Y - pred_Y) ** 2)

# 计算结构相似性指数
ssim_score = np.mean([ssim(test_Y[i], pred_Y[i], multichannel=True) for i in range(test_Y.shape[0])])

# 计算Dice系数
binary_test_Y = (test_Y > 0.5).astype(int)
binary_pred_Y = (pred_Y > 0.5).astype(int)
dice_coefficient = np.mean([f1_score(binary_test_Y[i].flatten(), binary_pred_Y[i].flatten()) for i in range(binary_test_Y.shape[0])])

# 计算交并比
iou = np.mean([jaccard_score(binary_test_Y[i].flatten(), binary_pred_Y[i].flatten()) for i in range(binary_test_Y.shape[0])])

print(f"Mean Squared Error: {mse}")
print(f"Structural Similarity Index: {ssim_score}")
print(f"Dice Coefficient: {dice_coefficient}")
print(f"Intersection over Union: {iou}")



# 选择要显示的图像
image_index = 0

# 创建一个画布，并添加三个子图
plt.figure(figsize=(12, 4))  # 设置画布大小

# 显示输入图像
plt.subplot(1, 3, 1)  # 1行3列，第1个子图
plt.imshow(test_X[image_index, :, :, :])
plt.title("Input Image")
plt.axis('off')

# 显示真实标签
plt.subplot(1, 3, 2)  # 1行3列，第2个子图
plt.imshow(test_Y[image_index, :, :, :])
plt.title("True Mask")
plt.axis('off')

# 显示模型的预测结果
plt.subplot(1, 3, 3)  # 1行3列，第3个子图
plt.imshow(pred_Y[image_index, :, :, :])
plt.title("Predicted Mask")
plt.axis('off')

plt.show()
