import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

IMG_SIZE = 28
PROCESSED_DATA_DIR = 'data_processed'


def load_data():
    """
    Betölti az előfeldolgozott .npy fájlokat, és előkészíti a CNN modell számára.
    Normalizál, 4D-s alakra hoz, "one-hot" kódol, és szétválasztja az adatokat.
    """
    try:
        X_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'))
        y_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_labels.npy'))
    except FileNotFoundError:
        print(f"HIBA: Nem találhatók a feldolgozott .npy fájlok a '{PROCESSED_DATA_DIR}' mappában.")
        print("Kérlek, futtasd először a 'src/data_preprocessing.py' szkriptet!")
        return None

    # Normalizálás
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    # Osztályok számának meghatározása
    num_classes = len(np.unique(y_train_full))

    # Átalakítás CNN formátumra: (db, magasság, szélesség, csatorna)
    X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Címkék One-Hot kódolása (pl. 5 -> [0,0,0,0,0,1,0,...])
    y_train_full_cat = tf.keras.utils.to_categorical(y_train_full, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    # Tanító adatok szétválasztása tanító és validációs halmazra (pl. 80-20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full_cat, test_size=0.2, random_state=42
    )

    # Visszaadunk mindent, amire szükségünk lehet
    return (X_train, y_train), (X_val, y_val), (X_test, y_test_cat), num_classes, y_test