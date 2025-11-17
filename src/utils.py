import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

IMG_SIZE = 28
PROCESSED_DATA_DIR = 'data_processed'


def load_data_for_training_and_prediction():
    """
    Betölti a tanító adatokat (és szétválasztja) ÉS a címke nélküli
    teszt adatokat a predikcióhoz.
    """
    try:
        # Tanító adatok betöltése
        X_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'))
        y_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy'))

        # Címke nélküli teszt adatok betöltése
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'))
        test_filenames = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_filenames.npy'))
    except FileNotFoundError as e:
        print(f"HIBA: Nem találhatók a feldolgozott .npy fájlok: {e}")
        print("Kérlek, futtasd először a 'src/data_preprocessing.py' szkriptet!")
        return None

    if X_train_full.size == 0 or X_test.size == 0:
        print(f"HIBA: A betöltött adatok üresek. Ellenőrizd a mappákat és futtasd újra a 'data_preprocessing.py'-t.")
        return None

    # --- Tanító adatok előkészítése ---
    X_train_full = X_train_full / 255.0
    num_classes = len(np.unique(y_train_full))
    X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)

    # --- Teszt adatok előkészítése ---
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return (X_train, y_train), (X_val, y_val), X_test, num_classes, test_filenames