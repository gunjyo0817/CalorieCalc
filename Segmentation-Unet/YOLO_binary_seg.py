import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tifffile import imread 
import cv2
import glob
from sklearn.preprocessing import LabelEncoder
import argparse


import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Constants
IMG_WIDTH, IMG_HEIGHT = 256, 256  # Input size for U-Net

def yolo_to_pixel_coords(yolo_box, img_width, img_height, padding=0):
    """
    Convert YOLO relative coordinates to pixel coordinates and add padding.
    """
    class_id, x_center, y_center, width, height = yolo_box
    x_center, y_center, width, height = (
        float(x_center) * img_width,
        float(y_center) * img_height,
        float(width) * img_width,
        float(height) * img_height,
    )
    x_min = int(x_center - width / 2) - padding
    y_min = int(y_center - height / 2) - padding
    x_max = int(x_center + width / 2) + padding
    y_max = int(y_center + height / 2) + padding

    # Ensure the coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    return class_id, x_min, y_min, x_max, y_max



def crop_image_and_mask(image, mask, box_coords):
    """
    Crop the image and mask using bounding box coordinates and class_id.
    """
    class_id, x_min, y_min, x_max, y_max = box_coords
    
    # Ensure the coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    
    # Crop image
    cropped_image = image[y_min:y_max, x_min:x_max]
    print(f"mask shape: {mask.shape}")
    
    # Extract and crop the mask for the given class_id
    if len(mask.shape) == 3:  # Multi-channel mask
        cropped_mask = mask[y_min:y_max, x_min:x_max, int(class_id)+1]
    else:
        # If mask is single channel, treat it as binary
        cropped_mask = mask[y_min:y_max, x_min:x_max]

    return cropped_image, cropped_mask


def load_yolo_boxes(yolo_txt_path, img_width, img_height):
    """
    Load YOLO boxes from a text file.
    """
    boxes = []
    with open(yolo_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            boxes.append(yolo_to_pixel_coords(parts, img_width, img_height, padding=0))
    return boxes

def prepare_training_data(image_paths, mask_paths, yolo_txt_paths):
    """
    Prepare U-Net training data from YOLO boxes and masks.
    """
    train_images = []
    train_masks = []
    
    for img_path, mask_path, yolo_txt_path in zip(image_paths, mask_paths, yolo_txt_paths):
        # Load image and mask
        image = cv2.imread(img_path)
        mask = imread(mask_path)  # Assuming mask has shape (height, width, num_classes)

        # Get image dimensions
        img_width, img_height = image.shape[1], image.shape[0]
        
        # Load YOLO boxes (using the specific image dimensions)
        yolo_boxes = load_yolo_boxes(yolo_txt_path, img_width, img_height)
        
        # Crop image and mask for each box
        for box in yolo_boxes:
            cropped_image, cropped_mask = crop_image_and_mask(image, mask, box)

            if cropped_image is None or cropped_mask is None:
                print(f"Skipping invalid crop for box {box}.")
                continue
            
            # Ensure cropped regions are non-empty before resizing
            if cropped_image.size == 0 or cropped_mask.size == 0:
                print(f"Skipping empty crop for box {box}.")
                continue
            
            # Resize to U-Net input size
            cropped_image = cv2.resize(cropped_image, (256, 256))
            cropped_mask = cv2.resize(cropped_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Normalize image and prepare mask
            train_images.append(cropped_image / 255.0)  # Normalize to [0, 1]
            train_masks.append((cropped_mask > 0).astype(np.float32))  # Binary mask for the class
    
    return np.array(train_images), np.expand_dims(np.array(train_masks), axis=-1)

def main(args):
    # Paths to data
    image_dir = '../data/train/images'
    mask_dir = '../data/train/masks'
    yolo_dir = '../data/train/yolo_boxes'

    # Load file paths
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png") + glob.glob(f"{image_dir}/*.jpeg"))
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
    yolo_txt_paths = sorted([os.path.join(yolo_dir, f) for f in os.listdir(yolo_dir) if f.endswith('.txt')])

    # Extract file names without extensions
    mask_filenames = {os.path.splitext(os.path.basename(path))[0] for path in mask_paths}
    
    # Filter images and YOLO files to include only those with a corresponding mask
    image_paths = [img for img in image_paths if os.path.splitext(os.path.basename(img))[0] in mask_filenames]
    yolo_txt_paths = [yolo for yolo in yolo_txt_paths if os.path.splitext(os.path.basename(yolo))[0] in mask_filenames]
    
    # Print results
    print(f"Number of images: {len(image_paths)}")
    print(f"Number of YOLO files: {len(yolo_txt_paths)}")
    print(f"Number of masks: {len(mask_paths)}")
    
    # Prepare training data
    print("Preparing training data...")
    train_images, train_masks = prepare_training_data(image_paths, mask_paths, yolo_txt_paths)
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_masks, test_size=0.2, random_state=42
    )
    
    print(f"Training images shape: {train_images.shape}")
    print(f"Training masks shape: {train_masks.shape}")

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Define U-Net model
    print("Building model...")
    model = sm.Unet(args.backbone, classes=1, activation='sigmoid', input_shape=(args.img_height, args.img_width, 3), encoder_weights='imagenet', encoder_freeze=True)
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[sm.metrics.iou_score],
    )

    # Train the model
    print("Training model...")
    model.fit(
        train_images, train_masks,
        validation_split=0.1,
        batch_size=args.batch_size,
        epochs=1,
        callbacks=[early_stopping],
    )
    
    set_trainable(model, recompile=False)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[sm.metrics.iou_score],
    )

    model.fit(
        train_images, train_masks,
        validation_split=0.1,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )
    
    # Save the model
    model.save('./weight/unet_yolo_earlystop_' + args.backbone + '_' + str(args.batch_size) + 'batch' + '.h5')
    print(f"Model saved as 'unet_yolo_earlystop_{args.backbone}_{str(args.batch_size)}batch.h5'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train U-Net model with YOLO annotations.")
    parser.add_argument('--img_width', type=int, default=256, help='Width of input images.')
    parser.add_argument('--img_height', type=int, default=256, help='Height of input images.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Backbone for U-Net.')

    args = parser.parse_args()
    main(args)
