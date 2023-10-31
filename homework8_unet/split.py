import os
import random
import shutil

def partition_images(folder_path, train_ratio, test_ratio, val_ratio):
    # Create train, test, and val folders
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')
    val_folder = os.path.join(folder_path, 'val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Get a list of image files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Shuffle the image files randomly
    # random.shuffle(image_files)

    # Calculate the number of images for each partition
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    test_count = int(total_images * test_ratio)
    val_count = total_images - train_count - test_count

    # Move images to the corresponding partitions
    for i, image_file in enumerate(image_files):
        src_path = os.path.join(folder_path, image_file)
        if i < train_count:
            dst_path = os.path.join(train_folder, image_file)
        elif i < train_count + test_count:
            dst_path = os.path.join(test_folder, image_file)
        else:
            dst_path = os.path.join(val_folder, image_file)
        shutil.move(src_path, dst_path)

    print("Successfully partitioned images!")

# Specify the folder path where the images are located
folder_path = r'D:\deeplearning\datasets\unet\imgs2'

# Specify the ratios for train, test, and val partitions
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Call the function to partition the images
partition_images(folder_path, train_ratio, test_ratio, val_ratio)