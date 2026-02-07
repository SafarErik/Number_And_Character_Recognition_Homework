import os
import numpy as np
import tensorflow as tf
from keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import argparse
import pandas as pd
import datetime
import time

from utils import load_data_for_training_and_prediction as load_data, morphological_augmentation
from models import build_simple_cnn, build_advanced_cnn, build_keras_mlp, build_hybrid_cnn, build_pro_hybrid_cnn, build_regularized_hybrid_cnn, build_deep_hybrid_cnn
from visualize import save_history_plot, save_misclassified_plot, save_confusion_matrix_plot

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth enabled ({len(gpus)} GPU)")
    except RuntimeError as e:
        print(e)

if gpus:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✅ Mixed Precision (float16) enabled")


# --- 1. Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train Character Recognition Model.')
    parser.add_argument('--model', type=str, default='advanced',
                        choices=['simple', 'advanced', 'mlp', 'hybrid', 'pro_hybrid', 'regularized', 'resnet', 'deep_hybrid'],
                        help='Model type to use (default: advanced)')

    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of the experiment (this will be the subfolder name).')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    return parser.parse_args()


def format_time(seconds):
    """Formats time into HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    args = parse_args()

    BASE_RESULTS_DIR = 'results'

    if args.run_name:
        RUN_NAME = args.run_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_NAME = f"{args.model}_{timestamp}"

    RUN_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, RUN_NAME)
    os.makedirs(RUN_RESULTS_DIR, exist_ok=True)

    print(f"\n--- STARTING NEW RUN ---")
    print(f"Experiment Name: {RUN_NAME}")
    print(f"All results will be saved to: {RUN_RESULTS_DIR}")

    # --- 2. Load Data ---
    print("Loading data...")
    data = load_data()
    if data is None: return

    (X_train, y_train), (X_val, y_val, y_val_labels), X_test, num_classes, test_filenames = data
    input_shape = X_train.shape[1:]

    # --- 3. Build Model ---
    if args.model == 'simple':
        model = build_simple_cnn(input_shape, num_classes)
    elif args.model == 'advanced':
        model = build_advanced_cnn(input_shape, num_classes)
    elif args.model == 'mlp':
        model = build_keras_mlp(input_shape, num_classes)
    elif args.model == 'hybrid':
        model = build_hybrid_cnn(input_shape, num_classes)
    elif args.model == 'pro_hybrid':
        model = build_pro_hybrid_cnn(input_shape, num_classes)
    elif args.model == 'regularized':
        model = build_regularized_hybrid_cnn(input_shape, num_classes)
    elif args.model == 'deep_hybrid':
        model = build_deep_hybrid_cnn(input_shape, num_classes)
    else:
        # Fallback for models not yet added to this if-else block but in choices
        print(f"Model {args.model} implementation not found in main loop.")
        return

    model.summary()

    # --- 4. Callbacks ---
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )

    model_checkpoint_path = os.path.join(RUN_RESULTS_DIR, "best_model.keras")
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy',
                                       save_best_only=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.00001, verbose=1)

    callbacks_list = [early_stopper, model_checkpoint, reduce_lr]

    # --- 5. Training with Timing ---
    print(f"\n--- Starting Training ({args.epochs} epochs) ---")
    training_start_time = time.time()

    if args.no_augmentation or args.model == 'mlp':
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                            validation_data=(X_val, y_val), callbacks=callbacks_list)
    else:
        print("Data Augmentation ENABLED.")

        # Default settings (e.g., for Hybrid model)
        aug_config = {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.0,
            'shear_range': 0.1,
            'fill_mode': 'constant',
            'cval': 0
        }

        if "shape_expert" in RUN_NAME:
            print(">> SPECIAL MODE: Shape Expert (HARD Augmentation!)")
            aug_config['zoom_range'] = 0.25
            aug_config['shear_range'] = 0.25
            aug_config['rotation_range'] = 25

        # If 'size_expert' --> Zoom off
        if "size_expert" in RUN_NAME:
            print(">> SPECIAL MODE: Size Expert (Zoom disabled!)")
            aug_config['zoom_range'] = 0.0

        # By default no morphological augmentation
        preprocessing_func = None

        if "thickness_expert" in RUN_NAME:
            print(">> SPECIAL MODE: Thickness Expert (Morphological Augmentation!)")
            # Disable zoom here to focus on thickness
            aug_config['zoom_range'] = 0.0
            preprocessing_func = morphological_augmentation

        datagen = ImageDataGenerator(
            **aug_config,
            preprocessing_function=preprocessing_func)
        datagen.fit(X_train)

        history = model.fit(datagen.flow(X_train, y_train, batch_size=args.batch_size),
                            epochs=args.epochs, validation_data=(X_val, y_val),
                            callbacks=callbacks_list)

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    actual_epochs = len(history.history['loss'])
    avg_time_per_epoch = total_training_time / actual_epochs

    print("\n--- Training Finished ---")
    print(f"Total training time: {format_time(total_training_time)} ({total_training_time:.2f} seconds)")
    print(f"Average time per epoch: {format_time(avg_time_per_epoch)} ({avg_time_per_epoch:.2f} seconds)")
    print(f"Actual number of epochs: {actual_epochs}")

    print(f"Loading best model from: {model_checkpoint_path}")
    model = tf.keras.models.load_model(model_checkpoint_path)

    # --- 6. PREDICTION ---
    print("\n--- Starting Final Prediction on TEST data ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("Predictions complete.")

    # --- 7. Save Results ---
    print(f"Saving results to '{RUN_RESULTS_DIR}'...")

    # Submission file
    submission_df = pd.DataFrame({'class': y_pred, 'TestImage': test_filenames})
    submission_path = os.path.join(RUN_RESULTS_DIR, "submission.csv")
    submission_df.to_csv(submission_path, sep=';', index=False)
    print(f"Submission file saved: {submission_path}")

    # Plots
    save_history_plot(history, os.path.join(RUN_RESULTS_DIR, "history.png"))
    save_misclassified_plot(model, X_val, y_val_labels,
                            os.path.join(RUN_RESULTS_DIR, "misclassified.png"))

    print("Generating validation report and confusion matrix...")
    try:
        # Predict on validation data first
        val_pred_probs = model.predict(X_val, verbose=0)
        val_pred = np.argmax(val_pred_probs, axis=1)

        # Save Report
        report = classification_report(y_val_labels, val_pred)
        report_path = os.path.join(RUN_RESULTS_DIR, "validation_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"--- Model information ---\n")
            f.write(f"Model type: {args.model}\n")
            f.write(f"Run name: {RUN_NAME}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Data augmentation: {'OFF' if args.no_augmentation or args.model == 'mlp' else 'ON'}\n")
            f.write(f"\n--- Validation results ---\n")
            f.write(report)
            f.write(f"\n--- Training statistics ---\n")
            f.write(f"Total training time: {format_time(total_training_time)} ({total_training_time:.2f} seconds)\n")
            f.write(f"Average time per epoch: {format_time(avg_time_per_epoch)} ({avg_time_per_epoch:.2f} seconds)\n")
            f.write(f"Number of epochs: {actual_epochs}\n")
        print(f"Validation report saved: {report_path}")

        # Save Confusion Matrix
        cm_path = os.path.join(RUN_RESULTS_DIR, "confusion_matrix.png")
        save_confusion_matrix_plot(y_val_labels, val_pred, cm_path)

    except Exception as e:
        print(f"Validation evaluation failed: {e}")

    print("\n--- Process Completed ---")


if __name__ == "__main__":
    main()