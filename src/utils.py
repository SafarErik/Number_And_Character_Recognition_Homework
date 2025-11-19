import numpy as np
from sklearn.model_selection import train_test_split
import os

IMG_SIZE = 32
PROCESSED_DATA_DIR = 'data_processed'


def load_data_for_training_and_prediction():
    """
    Betölti a data_preprocessing.py által generált .npy fájlokat.

    Returns:
        X_train, X_val, y_train_labels, y_val_labels, X_test, test_filenames, num_classes
        vagy None ha hiba történt
    """
    try:
        print("Adatok betöltése...")
        X_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'))
        y_train_full = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'))
        test_filenames = np.load(os.path.join(PROCESSED_DATA_DIR, 'test_filenames.npy'))
    except FileNotFoundError as e:
        print(f"HIBA: Nem találhatók a feldolgozott .npy fájlok! Futtasd le először a data_preprocessing.py-t!")
        print(f"Részletek: {e}")
        return None

    if X_train_full.size == 0 or X_test.size == 0:
        print(f"HIBA: A betöltött adatok üresek.")
        return None

    # Osztályok számának meghatározása
    max_label = np.max(y_train_full)
    num_classes = max_label + 1
    print(f"Sikeres betöltés. Osztályok száma: {num_classes}")

    # Normalizálás (0-255 -> 0.0-1.0)
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Csatorna dimenzió ellenőrzése (legyen (N, 32, 32, 1))
    if len(X_train_full.shape) == 3:
        X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    if len(X_test.shape) == 3:
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Validációs halmaz leválasztása (10%)
    # Fontos a stratify=y_train_full, hogy minden betűből jusson a validációba is
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    return X_train, X_val, y_train_labels, y_val_labels, X_test, test_filenames, num_classes
