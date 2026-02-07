import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def save_history_plot(history, file_path):
    """
    Saves training and validation curves to an image file.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy Plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        # Loss Plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        plt.savefig(file_path)
        print(f"Training curves saved: {file_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving curves: {e}")


def save_confusion_matrix_plot(y_true, y_pred, file_path):
    """
    Saves the confusion matrix to an image file.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.savefig(file_path, dpi=300)
        print(f"Confusion matrix saved: {file_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")


def save_misclassified_plot(model, X_val, y_val_true_labels, file_path, num_images=25):
    """
    Selects random misclassified images from the validation set
    and saves them to a plot.
    """
    print("Searching for misclassified predictions on validation data...")
    y_pred_probs = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    misclassified_indices = np.where(y_pred_labels != y_val_true_labels)[0]

    if len(misclassified_indices) == 0:
        print("Congratulations! The model made no errors on the validation data.")
        return

    num_to_sample = min(num_images, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_to_sample, replace=False)

    rows = int(np.ceil(num_to_sample / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows + 3))
    axes = axes.flatten()

    # Dynamic image size determination
    # X_val shape: (Batch, Height, Width, Channels)
    img_size = X_val.shape[1]

    for i, idx in enumerate(selected_indices):
        img = (X_val[idx].reshape(img_size, img_size) * 255).astype(np.uint8)

        true_label = y_val_true_labels[idx]
        pred_label = y_pred_labels[idx]

        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("Examples of Misclassified Predictions", fontsize=16, y=1.03)
    plt.savefig(file_path, bbox_inches='tight')
    print(f"Misclassified predictions plot saved: {file_path}")
    plt.close()