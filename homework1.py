import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("图像浏览器")
        self.root.geometry("1000x800")  # 设置窗口大小为1000x800

        self.initial_image = None

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.photo = None
        self.filename = None

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
        edit_menu.add_command(label="旋转", command=self.rotate_image)
        edit_menu.add_command(label="缩小", command=self.scale_down_image)
        edit_menu.add_command(label="放大", command=self.scale_up_image)
        edit_menu.add_command(label="反色", command=self.invert_colors)
        edit_menu.add_command(label="灰度化", command=self.grayscale)
        edit_menu.add_command(label="恢复", command=self.reset_image)
        menu_bar.add_cascade(label="编辑", menu=edit_menu)

    def create_widgets(self):
        self.canvas.bind("<Configure>", self.update_image)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.filename = file_path
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img)
        self.initial_image = self.image.copy()
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def rotate_image(self):
        if self.image:
            image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 逆时针旋转90度
            self.image = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def scale_up_image(self):
        self.scale_image(factor=1.5)  # 设置放大的比例

    def scale_down_image(self):
        self.scale_image(factor=0.75)  # 设置缩小的比例

    def scale_image(self, factor):
        if self.image:
            width, height = self.image.size
            new_width = int(width * factor)
            new_height = int(height * factor)

            image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            resized_image_cv = cv2.resize(image_cv, (new_width, new_height))
            self.image = Image.fromarray(cv2.cvtColor(resized_image_cv, cv2.COLOR_BGR2RGB))

            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def invert_colors(self):
        if self.image:
            image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            inverted_image_cv = cv2.bitwise_not(image_cv)
            self.image = Image.fromarray(cv2.cvtColor(inverted_image_cv, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def grayscale(self):
        if self.image:
            self.image = self.image.convert("L")
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def reset_image(self):
        if self.initial_image:
            self.image = self.initial_image.copy()
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def update_image(self, event):
        if self.filename:
            self.load_image(self.filename)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
