import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class ImageViewer:
    def __init__(self, root):


        self.root = root
        self.root.title("图像浏览器")
        self.root.geometry("2000x1500")  # 设置窗口大小
        # 创建左侧画布和标题
        self.canvas_left = tk.Canvas(self.root, width=700, height=500)
        self.canvas_left.grid(row=1, column=0, padx=10, pady=10)
        self.label_left = tk.Label(self.root, text="自定义")
        self.label_left.grid(row=0, column=0, padx=10, pady=10)

        # 创建右侧画布和标题
        self.canvas_right = tk.Canvas(self.root, width=700, height=500)
        self.canvas_right.grid(row=1, column=1, padx=10, pady=10)
        self.label_right = tk.Label(self.root, text="opencv")
        self.label_right.grid(row=0, column=1, padx=10, pady=10)

        self.image = None
        self.right_image = None
        self.original_image = None
        self.opencv_histogram = None
        self.custom_histogram = None
        self.filter_size = None
        # self.histogram_image = None
        self.photo_left = None
        self.photo_right = None
        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="打开文件", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menu_bar.add_cascade(label="文件", menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="放大", command=self.scale_up)
        edit_menu.add_command(label="斜切", command=self.shear)
        edit_menu.add_command(label="透视", command=self.perspective)
        menu_bar.add_cascade(label="双线性插值", menu=edit_menu)

        edit_menu2 = tk.Menu(menu_bar, tearoff=0)
        edit_menu2.add_command(label="均衡化", command=self.histogram_equalization)
        edit_menu2.add_command(label="计算灰度直方图", command=self.calculate_histogram)
        menu_bar.add_cascade(label="直方图", menu=edit_menu2)

        edit_menu3 = tk.Menu(menu_bar, tearoff=0)
        edit_menu3.add_command(label="添加噪音", command=self.add_noisy)
        edit_menu3.add_command(label="滤波处理", command=self.median_filter)
        menu_bar.add_cascade(label="中值滤波", menu=edit_menu3)

        edit_menu4 = tk.Menu(menu_bar, tearoff=0)
        edit_menu4.add_command(label="使用robert算子", command=lambda: self.sharpen_image(operator='laplace'))
        edit_menu4.add_command(label="使用sobel算子", command=lambda: self.sharpen_image(operator='sobel'))
        edit_menu4.add_command(label="使用laplace算子", command=lambda: self.sharpen_image(operator='laplace'))
        menu_bar.add_cascade(label="图像锐化", menu=edit_menu4)

        edit_menu5 = tk.Menu(menu_bar, tearoff=0)
        edit_menu5.add_command(label="低通处理", command=lambda: self.filter_image(filter_type='lowpass'))
        edit_menu5.add_command(label="高通处理", command=lambda: self.filter_image(filter_type='highpass'))
        menu_bar.add_cascade(label="频域处理", menu=edit_menu5)

    def create_widgets(self):
        self.canvas_left.bind("<Button-1>", self.left_canvas_click)

    def open_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if self.file_path:
            self.load_image()

    def load_image(self):
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        img = cv2.imread(self.file_path)
        height, width, _ = img.shape
        # 设置目标宽度和高度
        target_width = 700
        target_height = 500
        # 计算缩放比例
        if width > target_width or height > target_height:
            if width / target_width > height / target_height:
                scale_factor = width / target_width
            else:
                scale_factor = height / target_height
            # 缩放图像
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            img = cv2.resize(img, (new_width, new_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img
        self.original_image = img.copy()
        self.update_canvas()

    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.update_canvas()

    def update_canvas(self):
        if self.image is not None:
            self.photo_left = ImageTk.PhotoImage(Image.fromarray(self.image))
            self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.photo_left)

        if self.right_image is not None:
            self.photo_right = ImageTk.PhotoImage(Image.fromarray(self.right_image))
            self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.photo_right)


    def left_canvas_click(self, event):
        self.scale_up()

    def scale_up(self):
        if self.image is not None:
            scale_factor = 1.2  # 可以根据需要调整放大倍数
            height, width, _ = self.image.shape
            new_height, new_width = int(height * scale_factor), int(width * scale_factor)
            new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            # opencv的放大
            self.right_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # 自定义放大
            for y in range(new_height):
                for x in range(new_width):
                    src_x = x / scale_factor
                    src_y = y / scale_factor
                    for c in range(3):
                        new_image[y, x, c] = self.bilinear_interpolation(self.image[:, :, c], src_x, src_y)
            self.image = new_image

            self.update_canvas()

    def shear(self):
        if self.image is not None:
            shear_x = 0.2  # 可以根据需要调整斜切参数
            shear_y = 0.1  # 可以根据需要调整斜切参数
            height, width, _ = self.image.shape
            new_image = np.zeros((height, width, 3), dtype=np.uint8)

            # opencv实现斜切
            # 计算仿射变换矩阵
            shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
            # 应用斜切变换
            self.right_image = cv2.warpAffine(self.image, shear_matrix, (width, height), flags=cv2.INTER_LINEAR)

            # 自定义实现斜切
            for y in range(height):
                for x in range(width):
                    src_x = x + shear_x * y
                    src_y = y + shear_y * x
                    if 0 <= src_x < width -1 and 0 <= src_y < height -1:
                        for c in range(3):
                            new_image[y, x, c] = self.bilinear_interpolation(self.image[:, :, c], src_x, src_y)
            self.image = new_image

            self.update_canvas()

    def perspective(self):
        if self.image is not None:
            perspective_matrix = np.array([[1, 0.2, 0], [0.1, 1, 0], [0, 0, 1]], dtype=np.float32)
            height, width, _ = self.image.shape
            new_image = np.zeros((height, width, 3), dtype=np.uint8)

            # cv
            # # 定义透视变换的四个点坐标（顺时针定义）
            # pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
            # # 定义透视后的四个点坐标
            # pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            # # 计算透视变换矩阵
            # perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

            # 应用透视变换
            # cv perspective
            # opencv实现透视
            self.right_image = cv2.warpPerspective(self.image, perspective_matrix, (width, height), flags=cv2.INTER_LINEAR)

            # 自定义实现透视
            for y in range(height):
                for x in range(width):
                    src_x = (perspective_matrix[0, 0] * x + perspective_matrix[0, 1] * y + perspective_matrix[0, 2]) / \
                            (perspective_matrix[2, 0] * x + perspective_matrix[2, 1] * y + perspective_matrix[2, 2])
                    src_y = (perspective_matrix[1, 0] * x + perspective_matrix[1, 1] * y + perspective_matrix[1, 2]) / \
                            (perspective_matrix[2, 0] * x + perspective_matrix[2, 1] * y + perspective_matrix[2, 2])
                    if 0 <= src_x < width and 0 <= src_y < height:
                        for c in range(3):
                            new_image[y, x, c] = self.bilinear_interpolation(self.image[:, :, c], src_x, src_y)
            self.image = new_image

            self.update_canvas()

    def bilinear_interpolation(self, channel, x, y):
        height, width = channel.shape[:2]
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
        dx, dy = x - x1, y - y1

        interpolated_value = (1 - dx) * (1 - dy) * channel[y1, x1] + \
                             dx * (1 - dy) * channel[y1, x2] + \
                             (1 - dx) * dy * channel[y2, x1] + \
                             dx * dy * channel[y2, x2]
        return interpolated_value

    # 自定义函数计算灰度直方图
    def calculate_custom_histogram(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = np.zeros(256, dtype=np.int64)

        for pixel_value in np.ravel(gray_image):
            hist[pixel_value] += 1

        return hist

    # 使用OpenCV计算灰度直方图
    def calculate_opencv_histogram(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        return hist

    def calculate_histogram(self):
        self.load_image()
        # 计算灰度直方图
        if self.image is not None:
            self.custom_histogram = self.calculate_custom_histogram(self.image)
            self.opencv_histogram = self.calculate_opencv_histogram(self.image)

            # 自定义灰度直方图
            plt.figure(1, figsize=(12, 6))
            plt.title('Custom Histogram')
            plt.bar(range(256), self.custom_histogram, width=1.0, color='b')
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            self.image = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGBA2BGR)
            self.image = cv2.resize(self.image, (700, 500))


            # 使用OpenCV计算的灰度直方图
            plt.figure(2, figsize=(12, 6))
            plt.title('OpenCV Histogram')
            plt.plot(self.opencv_histogram, color='r')
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            self.right_image = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGBA2BGR)
            self.right_image = cv2.resize(self.right_image, (700, 500))

            self.update_canvas()

    def histogram_equalization(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # 使用OpenCV实现均匀化
            equalized_image_cv = cv2.equalizeHist(gray_image)
            self.right_image = equalized_image_cv

            # 使用自定义直方图均匀化
            if self.custom_histogram is None:
                self.custom_histogram = self.calculate_custom_histogram(self.original_image)
            # 计算灰度直方图和CDF
            hist = self.custom_histogram
            cdf = hist.cumsum()
            # 归一化CDF
            cdf_normalized = cdf * hist.max() / cdf.max()
            # 使用CDF进行均衡化
            equalized_image = np.interp(gray_image, range(256), cdf_normalized).astype(np.uint8)
            self.image = equalized_image

            self.update_canvas()

    def custom_median_filter(self, image, kernel_size):
        height, width = image.shape
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        pad = kernel_size // 2  # 卷积核的半径
        for i in range(pad, height - pad):
            for j in range(pad, width - pad):
                # 获取卷积核区域内的像素值
                neighborhood = image[i - pad: i + pad + 1, j - pad: j + pad + 1]
                # 计算中位数并将其赋给滤波后的像素
                filtered_image[i, j] = np.median(neighborhood)
        return filtered_image

    def median_filter(self):
        if self.image is not None:
            image = self.image
            # 将彩色图像拆分成通道
            r, g, b = cv2.split(image)
            kernel_size = 3  # 卷积核大小

            # 自定义中值滤波
            filtered_image_r = self.custom_median_filter(r, kernel_size)
            filtered_image_g = self.custom_median_filter(g, kernel_size)
            filtered_image_b = self.custom_median_filter(b, kernel_size)
            filtered_image = cv2.merge((filtered_image_r, filtered_image_g, filtered_image_b))
            self.image = filtered_image

            # OpenCV的中值滤波函数
            opencv_filtered_image_r = cv2.medianBlur(r, kernel_size)
            opencv_filtered_image_g = cv2.medianBlur(g, kernel_size)
            opencv_filtered_image_b = cv2.medianBlur(b, kernel_size)
            opencv_filtered_image = cv2.merge((opencv_filtered_image_r, opencv_filtered_image_g, opencv_filtered_image_b))
            self.right_image = opencv_filtered_image

            self.update_canvas()

    # 添加彩色椒盐噪声
    def add_color_salt_and_pepper_noise(self, image, salt_prob, pepper_prob):
        noisy_image = np.copy(image)
        total_pixels = image.size

        # 添加椒噪声
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[salt_coords[0], salt_coords[1], :] = [255, 255, 255]

        # 添加盐噪声
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = [0, 0, 0]

        return noisy_image

    # 添加彩色高斯噪声
    def add_color_gaussian_noise(self, image, mean=0, std=25):
        noisy_image = np.copy(image)
        gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(noisy_image, gaussian_noise)

        return noisy_image

    def add_noisy(self):
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")

        # 添加彩色椒盐噪声
        image = self.add_color_salt_and_pepper_noise(self.original_image, salt_prob=0.001, pepper_prob=0.01)

        # # 添加彩色高斯噪声
        # image = self.add_color_gaussian_noise(self.original_image, mean=0, std=5)
        self.image = image

        self.update_canvas()

    def sharpen_image(self, operator='laplace'):
        self.load_image()

        if operator == 'robert':
            kernel = np.array([[-1, 0], [0, 1]])
        elif operator == 'sobel':
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif operator == 'laplace':
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        else:
            raise ValueError("Invalid operator. Choose 'robert', 'sobel', or 'laplace'.")

        # 使用opencv的锐化
        sharpened_image = cv2.filter2D(self.image, -1, kernel)
        self.right_image = sharpened_image

        # 使用自定义的锐化
        sharpened_image_my = np.zeros_like(self.image, dtype=np.float32)
        height, width, _ = self.image.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                patch = self.image[y - 1:y + 2, x - 1:x + 2].astype(np.float32)
                result = np.sum(patch * kernel)
                sharpened_image_my[y, x] = result
        sharpened_image_my = np.clip(sharpened_image_my, 0, 255).astype(np.uint8)
        self.image = sharpened_image_my

        self.update_canvas()

    def filter_image(self, filter_type='lowpass'):
        self.canvas_right.delete("all")
        # 创建一个简单的对话框来输入 filter_size
        self.filter_size = simpledialog.askinteger("输入滤波器大小", "请输入滤波器大小：", initialvalue=30)
        if self.filter_size is None:
            self.filter_size = 30  # 默认值
        self.canvas_left.delete("all")
        # 进行傅里叶变换
        f_transform = np.fft.fft2(self.original_image, axes=(0, 1))
        f_transform_shifted = np.fft.fftshift(f_transform, axes=(0, 1))

        # 定义滤波器
        rows, cols, channels = self.original_image.shape
        crow, ccol = rows // 2, cols // 2  # 中心点坐标
        mask = None
        # 创建频域滤波器
        if filter_type == 'lowpass':
            mask = np.zeros((rows, cols, channels), np.uint8)
            mask[crow - self.filter_size:crow + self.filter_size, ccol - self.filter_size:ccol + self.filter_size, :] = 1
        elif filter_type == 'highpass':
            mask = np.ones((rows, cols, channels), np.uint8)
            mask[crow - self.filter_size:crow + self.filter_size, ccol - self.filter_size:ccol + self.filter_size, :] = 0

        # 应用滤波器
        f_transform_shifted_filtered = f_transform_shifted * mask

        # 进行逆傅里叶变换
        f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered, axes=(0, 1))
        filtered_image = np.fft.ifft2(f_transform_filtered, axes=(0, 1))
        filtered_image = np.abs(filtered_image).astype(np.uint8)
        self.image = filtered_image

        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
