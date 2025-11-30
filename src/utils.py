import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

IMG_SIZE = 32
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