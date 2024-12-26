import os
import shutil
import random

# dataset/
# ├── images/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── labels/
# │   ├── img1.txt
# │   ├── img2.txt
# │   └── ...

# Paths to your dataset
dataset_path = 'C:/Users/TUF/Desktop/Irene/ML/Project_test_change1_only_183'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Create directories for train, val, and test
train_images_path = os.path.join(dataset_path, 'train', 'images')
val_images_path = os.path.join(dataset_path, 'val', 'images')
test_images_path = os.path.join(dataset_path, 'test', 'images')
train_labels_path = os.path.join(dataset_path, 'train', 'labels')
val_labels_path = os.path.join(dataset_path, 'val', 'labels')
test_labels_path = os.path.join(dataset_path, 'test', 'labels')

def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

directories = [
    train_images_path, val_images_path,
    train_labels_path, val_labels_path
]

# Clear directories if not empty
for directory in directories:
    clear_directory(directory)

# Create directories again to ensure they exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

# Shuffle the dataset
random.shuffle(image_files)

# Split the dataset (70% train, 20% val, 10% test)
train_size = int(0.8 * len(image_files))
val_size = int(0.2 * len(image_files))
train_files = image_files[:train_size]
val_files = image_files[train_size:]
# test_files = image_files[train_size + val_size:]

# Function to move files to respective directories
def move_files(file_list, src_images, src_labels, dest_images, dest_labels):
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]
        img_src = os.path.join(src_images, file_name)
        label_src = os.path.join(src_labels, f'{base_name}.txt')

        img_dest = os.path.join(dest_images, file_name)
        label_dest = os.path.join(dest_labels, f'{base_name}.txt')

        shutil.copyfile(img_src, img_dest)
        shutil.copyfile(label_src, label_dest)

def split_files():
    # Move files to train, val, and test directories
    move_files(train_files, images_path, labels_path, train_images_path, train_labels_path)
    move_files(val_files, images_path, labels_path, val_images_path, val_labels_path)
    # move_files(test_files, images_path, labels_path, test_images_path, test_labels_path)

if (__name__ == "__main__"):
    split_files()