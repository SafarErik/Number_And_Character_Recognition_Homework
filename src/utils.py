import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2

IMG_SIZE = 64
PROCESSED_DATA_DIR = 'data_processed'


def load_data_for_training_and_prediction():
    try:
        X_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'))
        y_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'))
        test_filenames = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_filenames.npy'))
    except FileNotFoundError as e:
        print(f"ERROR: Processed .npy files not found: {e}")
        return None

    if X_train_full.size == 0 or X_test.size == 0:
        print(f"ERROR: Loaded data is empty.")
        return None

    # --- CLASS CALCULATION ---
    max_label = np.max(y_train_full)
    num_classes = max_label + 1
    print(f"Labels loaded. Highest ID: {max_label}. Number of classes: {num_classes}")

    # --- Prepare Training Data ---
    X_train_full = X_train_full / 255.0
    X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    # One-hot encoding
    y_train_cat = tf.keras.utils.to_categorical(y_train_labels, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val_labels, num_classes)

    # --- Prepare Test Data ---
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return (X_train, y_train_cat), (X_val, y_val_cat, y_val_labels), X_test, num_classes, test_filenames


def morphological_augmentation(image):
    """
    Randomly thickens (dilation) or thins (erosion) the lines.
    Called by Keras ImageDataGenerator for each image.
    Input: (64, 64, 1) float array (0.0 - 1.0)
    """
    # 1. Decision: Do something? (50% chance to stay original)
    if np.random.rand() < 0.5:
        return image

    # 2. Convert to 0-255 uint8 format (OpenCV likes this)
    img_uint8 = (image * 255).astype(np.uint8)

    # 3. Create Kernel (the "brush")
    # A 2x2 kernel makes subtle changes. 3x3 would be too strong.
    kernel = np.ones((2, 2), np.uint8)

    # 4. Choose random operation
    op_type = np.random.choice(["erode", "dilate"])

    if op_type == "erode":
        # Thinning (Erosion) - e.g. pencil effect
        img_aug = cv2.erode(img_uint8, kernel, iterations=1)
    else:
        # Thickening (Dilation) - e.g. marker effect
        img_aug = cv2.dilate(img_uint8, kernel, iterations=1)

    # 5. Convert back to 0-1 float format and 3D shape
    # Important: OpenCV sometimes removes the channel dimension, we must restore it!
    img_aug = img_aug.astype(np.float32) / 255.0

    # Ensure shape remains (64, 64, 1)
    if len(img_aug.shape) == 2:
        img_aug = np.expand_dims(img_aug, axis=-1)

    return img_aug