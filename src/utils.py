import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

IMG_SIZE = 64
PROCESSED_DATA_DIR = 'data_processed'


def load_data_for_training_and_prediction():
    try:
        X_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'))
        y_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'))
        test_filenames = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_filenames.npy'))
    except FileNotFoundError as e:
        print(f"HIBA: Nem találhatók a feldolgozott .npy fájlok: {e}")
        return None

    if X_train_full.size == 0 or X_test.size == 0:
        print(f"HIBA: A betöltött adatok üresek.")
        return None

    # --- OSZTÁLYSZÁMÍTÁS ---
    max_label = np.max(y_train_full)
    num_classes = max_label + 1
    print(f"Címkék betöltve. Legmagasabb ID: {max_label}. Osztályok száma: {num_classes}")

    # --- Tanító adatok előkészítése ---
    X_train_full = X_train_full / 255.0
    X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    # One-hot kódolás
    y_train_cat = tf.keras.utils.to_categorical(y_train_labels, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val_labels, num_classes)

    # --- Teszt adatok előkészítése ---
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Módosított visszatérés:
    return (X_train, y_train_cat), (X_val, y_val_cat, y_val_labels), X_test, num_classes, test_filenames


import cv2
import numpy as np


def morphological_augmentation(image):
    """
    Véletlenszerűen vastagítja (dilatáció) vagy vékonyítja (erózió) a vonalakat.
    A Keras ImageDataGenerator hívja meg minden képre külön-külön.
    Bemenet: (64, 64, 1) float tömb (0.0 - 1.0 között)
    """
    # 1. Döntés: Csináljunk valamit? (50% esély, hogy eredeti marad)
    if np.random.rand() < 0.5:
        return image

    # 2. Konvertálás 0-255 uint8 formátumra (az OpenCV ezt szereti)
    img_uint8 = (image * 255).astype(np.uint8)

    # 3. Kernel létrehozása (az "ecset")
    # Egy 2x2-es kernel finom változtatást csinál. 3x3 már nagyon durva lenne.
    kernel = np.ones((2, 2), np.uint8)

    # 4. Véletlen művelet kiválasztása
    op_type = np.random.choice(["erode", "dilate"])

    if op_type == "erode":
        # Vékonyítás (Erózió) - pl. ceruza effektus
        img_aug = cv2.erode(img_uint8, kernel, iterations=1)
    else:
        # Vastagítás (Dilatáció) - pl. filctoll effektus
        img_aug = cv2.dilate(img_uint8, kernel, iterations=1)

    # 5. Visszakonvertálás 0-1 float formátumra és 3D alakra
    # Fontos: Az OpenCV néha leveszi a csatorna dimenziót, ezt pótolni kell!
    img_aug = img_aug.astype(np.float32) / 255.0

    # Biztosítjuk, hogy a forma (64, 64, 1) maradjon
    if len(img_aug.shape) == 2:
        img_aug = np.expand_dims(img_aug, axis=-1)

    return img_aug