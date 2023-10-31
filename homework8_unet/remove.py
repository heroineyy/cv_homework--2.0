import os

def remove_mask_suffix(folder_path):
    # 获取指定文件夹中的文件列表
    file_list = os.listdir(folder_path)

    # 遍历文件列表
    for file_name in file_list:
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否是文件而不是文件夹
        if os.path.isfile(file_path):
            # 判断文件名中是否包含 "_mask" 字符串
            if "_mask.gif" in file_name:
                # 构造新的文件名，去掉 "_mask" 字符串
                new_file_name = file_name.replace("_mask.gif", ".jpg")

                # 构造新的文件路径
                new_file_path = os.path.join(folder_path, new_file_name)

                # 重命名文件
                os.rename(file_path, new_file_path)

    print("Successfully removed the '_mask' substring from image names!")

# 指定文件夹的路径
folder_path = r'D:\deeplearning\datasets\unet\masks2\train'

# 调用函数去除文件名中的 "_mask" 字符串
remove_mask_suffix(folder_path)