import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def save_history_plot(history, file_path):
    """
    Elmenti a tanítási és validációs görbéket egy képfájlba.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pontosság ábra
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        # Hiba ábra
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        plt.savefig(file_path)
        print(f"Tanítási görbék elmentve: {file_path}")
        plt.close()
    except Exception as e:
        print(f"Hiba a görbék mentésekor: {e}")

def save_confusion_matrix_plot(y_true, y_pred, file_path):
    """
    Elmenti a konfúziós mátrixot egy képfájlba.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(20, 16))  # Nagyobb méret a több osztály miatt
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.savefig(file_path, dpi=300)  # Nagyobb felbontás
        print(f"Konfúziós mátrix elmentve: {file_path}")
        plt.close()
    except Exception as e:
        print(f"Hiba a konfúziós mátrix mentésekor: {e}")