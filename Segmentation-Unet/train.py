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

import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
N_CLASSES = 37  # Number of segmentation classes
BATCH_SIZE = 32
EPOCHS = 200

from tensorflow.keras.utils import to_categorical

def load_data():
    # Load and sort mask paths first
    mask_paths = sorted(glob.glob("./data/train_masks/*.tif"))
    mask_filenames = {os.path.splitext(os.path.basename(path))[0] for path in mask_paths}
    
    # Initialize lists for images and masks
    train_images = []
    train_masks = []
    
    # Iterate through image paths for both .jpg and .png
    image_paths = sorted(glob.glob("data/train/images/*.jpg") + glob.glob("data/train/images/*.png"))
    
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Check if corresponding mask exists
        if img_name in mask_filenames:
            # Load and preprocess the image (RGB)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            train_images.append(img)
            
            # Load and preprocess the mask
            mask_path = f"./data/train_masks/{img_name}.tif"
            mask = imread(mask_path)
            
            if mask.ndim == 3 and mask.shape[-1] == N_CLASSES:  # Ensure it's a one-hot mask
                mask = np.argmax(mask, axis=-1)  # Convert to single-channel mask
            
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            train_masks.append(mask)
    
    # Convert lists to numpy arrays
    train_images = np.array(train_images, dtype=np.float32)
    train_masks = np.array(train_masks, dtype=np.int32)
    
    # Normalize images to range [0, 1]
    train_images /= 255.0
    
    # One-hot encode masks
    train_masks = np.eye(N_CLASSES)[train_masks]  # One-hot encode masks to shape [?, 256, 256, 50]

    print("train_images shape:", train_images.shape)
    print("train_masks shape:", train_masks.shape)
    
    unique_classes = np.unique(train_masks.argmax(axis=-1))  # Confirm unique classes
    print("Unique classes:", unique_classes)
    
    return train_images, train_masks


class DataGenerator(Sequence):
    """
    Custom Data Generator for image and mask data.
    """
    def __init__(self, image_gen, mask_gen, batch_size):
        self.image_gen = image_gen
        self.mask_gen = mask_gen
        self.batch_size = batch_size
    
    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.image_gen) / self.batch_size))
    
    def __getitem__(self, index):
        # Fetch the next batch from generators
        images = next(self.image_gen)
        masks = next(self.mask_gen)
        return images, masks



def augment_data(images, masks):
    """
    Perform data augmentation on images and masks.
    """
    data_gen_args = dict(
        # rotation_range=10,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Fit the generators (if needed)
    image_datagen.fit(images, augment=True)
    mask_datagen.fit(masks, augment=True)
    
    # Generate augmented data
    image_gen = image_datagen.flow(images, batch_size=BATCH_SIZE, seed=42)
    mask_gen = mask_datagen.flow(masks, batch_size=BATCH_SIZE, seed=42)
    
    return DataGenerator(image_gen, mask_gen, batch_size=BATCH_SIZE)

def main():
    # Load data
    images, masks = load_data()

    print("Number of images:", len(images))
    print("Number of masks:", len(masks))
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.1, random_state=42)
        
    # Preprocess input
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    
    # Data augmentation for training set
    train_generator = augment_data(X_train, y_train)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Define model
    model = sm.Unet('resnet101', classes=N_CLASSES, activation='softmax', encoder_weights='imagenet', encoder_freeze=True)
    model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[sm.metrics.iou_score],
    )

    # Fit model with data augmentation
    model.fit(
        # train_generator,
        X_train, y_train,
        # steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    
    # Release all layers for training
    set_trainable(model, recompile=False)
    model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[sm.metrics.iou_score],
    )

    # Continue training
    model.fit(
        # train_generator,
        X_train, y_train,
        # steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    model.save('unet_model_earlystop_resnet101_augmented.h5')

if __name__ == "__main__":
    main()


