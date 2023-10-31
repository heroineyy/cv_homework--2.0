import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class ImageViewer:
    def __init__(self, root):

        self.value = None
        self.file_path = None
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
        # 创建一个滚动体 动态改变
        self.scrollbar = ttk.Scale(self.root, from_=0, to=200, orient="horizontal", command=self.on_scroll)
        self.value_label = ttk.Label(root, text="")
        self.scrollbar.place(x=300, y=600)
        self.value_label.place(x=320, y=580)

        self.image = None
        self.right_image = None
        self.original_image = None

        self.filter_size = None
        self.photo_left = None
        self.photo_right = None
        self.create_menu()

    def on_scroll(self, *args):
        threshold = int(self.scrollbar.get())
        self.value_label.config(text=f"参数值: {threshold}")
        self.houghLines(threshold)

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="打开文件", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menu_bar.add_cascade(label="文件", menu=file_menu)

        edit_menu6 = tk.Menu(menu_bar, tearoff=0)
        edit_menu6.add_command(label="膨胀", command=self.erosion)
        edit_menu6.add_command(label="腐蚀", command=self.dilation)
        menu_bar.add_cascade(label="腐蚀和膨胀", menu=edit_menu6)

        edit_menu7 = tk.Menu(menu_bar, tearoff=0)
        edit_menu7.add_command(label="canny算子", command=self.canny_edge)
        menu_bar.add_cascade(label="检测鱼骨边缘", menu=edit_menu7)

        edit_menu8 = tk.Menu(menu_bar, tearoff=0)
        edit_menu8.add_command(label="霍夫变换", command=lambda: self.houghLines(threshold=30))
        menu_bar.add_cascade(label="检测直线", menu=edit_menu8)

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

    def erosion(self):
        self.reset_image()
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=np.uint8)
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            height, width = image.shape
            kh, kw = kernel.shape
            h, w = kh // 2, kw // 2
            result = np.zeros((height, width), dtype=np.uint8)
            for i in range(h, height - h):
                for j in range(w, width - w):
                    result[i, j] = np.min(image[i - h:i + h + 1, j - w:j + w + 1] * kernel)
            # opencv实现腐蚀
            self.right_image = cv2.erode(image, kernel, iterations=1)
            self.image = result

            self.update_canvas()

    def dilation(self):
        self.reset_image()
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            height, width = image.shape
            kh, kw = kernel.shape
            h, w = kh // 2, kw // 2
            result = np.zeros((height, width), dtype=np.uint8)
            for i in range(h, height - h):
                for j in range(w, width - w):
                    result[i, j] = np.max(image[i - h:i + h + 1, j - w:j + w + 1] * kernel)
            # opencv实现膨胀
            self.right_image = cv2.dilate(image, kernel, iterations=1)
            self.image = result
            self.update_canvas()

    def canny_edge(self):
        # 读取包含鱼骨的图像
        self.reset_image()
        image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # 调整Canny算子的参数，根据实际情况进行调优
        low_threshold = 50
        high_threshold = 150

        # 使用Canny算子进行边缘检测
        result = cv2.Canny(image, low_threshold, high_threshold)
        self.image = result
        self.update_canvas()

    def houghLines(self, threshold=30):
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        image = cv2.imread('data/chair.bmp')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 50, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=50, maxLineGap=10)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.image = image
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
