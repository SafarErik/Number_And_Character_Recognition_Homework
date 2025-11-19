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


def save_misclassified_plot(model, X_val, y_val_true_labels, file_path, num_images=25):
    """
    Kiválaszt véletlenszerűen elrontott képeket a validációs halmazból
    és elmenti őket egy ábrára.

    Args:
        model: A betanított Keras modell.
        X_val: A validációs képadatok (normalizálva, 4D formátumban).
        y_val_true_labels: A validációs címkék (NEM one-hot, sima számok, pl. 19, 45).
        file_path: A mentés helye.
        num_images: Hány képet mutasson.
    """
    print("Elrontott jóslatok keresése a validációs adatokon...")
    # Jóslatok készítése a validációs adatokra
    y_pred_probs = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)  # A jósolt indexek

    # Az elrontott képek indexeinek megkeresése
    misclassified_indices = np.where(y_pred_labels != y_val_true_labels)[0]

    if len(misclassified_indices) == 0:
        print("Gratulálok! A modell nem hibázott a validációs adatokon.")
        return

    # Véletlenszerű mintavétel az elrontottakból
    num_to_sample = min(num_images, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_to_sample, replace=False)

    # Ábra előkészítése (5 oszlop)
    rows = int(np.ceil(num_to_sample / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows + 3))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        # Kép visszaalakítása 2D-be és 0-255 skálára
        img = (X_val[idx].reshape(32, 32) * 255).astype(np.uint8)

        true_label = y_val_true_labels[idx]
        pred_label = y_pred_labels[idx]

        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Valós: {true_label}\nJósolt: {pred_label}", color='red')
        ax.axis('off')

    # Üres ábrák elrejtése
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0)
    plt.suptitle("Példák elrontott jóslatokra", fontsize=16, y=1.03)
    plt.savefig(file_path)
    print(f"Elrontott jóslatok ábrája elmentve: {file_path}")
    plt.close()