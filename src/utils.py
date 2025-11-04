import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 28


def load_data(model_type='cnn'):
    """
    Betölti az előfeldolgozott .npy fájlokat és előkészíti őket.
    model_type: 'cnn' vagy 'mlp' (ez dönti el a reshape-et)
    """
    X_train_full = np.load('data_processed/train_features.npy')
    y_train_full = np.load('data_processed/train_labels.npy')
    X_test = np.load('data_processed/test_features.npy')
    y_test = np.load('data_processed/test_labels.npy')

    # Normalizálás
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    # Osztályok számának meghatározása
    num_classes = len(np.unique(y_train_full))

    if model_type == 'cnn':
        # 4D-re alakítás CNN-hez: (db, mag, szél, csat)
        X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        # One-hot kódolás
        y_train_full = tf.keras.utils.to_categorical(y_train_full, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Készítünk egy belső validációs halmazt a tanító adatokból
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes


def save_evaluation(history, model, X_test, y_test_labels, model_name="model"):
    """
    Elmenti a kiértékelő ábrákat és riportokat a 'results/' mappába.
    """
    # ... (ide jöhet a history plot-oló kód) ...
    plt.savefig(f'results/{model_name}_accuracy_loss.png')

    # ... (ide jöhet a konfúziós mátrix kódja) ...
    plt.savefig(f'results/{model_name}_confusion_matrix.png')

    # ... (classification report mentése fájlba) ...
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test_labels, y_pred)
    with open(f'results/{model_name}_report.txt', 'w') as f:
        f.write(report)

    print(f"Kiértékelés elmentve a 'results/{model_name}' néven.")