import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import argparse
import pandas as pd
import datetime
import time

from utils import load_data_for_training_and_prediction as load_data
from models import build_simple_cnn, build_advanced_cnn, build_keras_mlp, build_hybrid_cnn, build_pro_hybrid_cnn, build_regularized_hybrid_cnn
from visualize import save_history_plot, save_misclassified_plot


# --- 1. Konfiguráció ---
def parse_args():
    parser = argparse.ArgumentParser(description='Karakterfelismerő modell tanítása.')
    parser.add_argument('--model', type=str, default='advanced',
                        choices=['simple', 'advanced', 'mlp', 'hybrid', 'pro_hybrid', 'regularized'],
                        help='A használni kívánt modell típusa (default: advanced)')

    parser.add_argument('--run_name', type=str, default=None,
                        help='A kísérlet neve (ez lesz az alkönyvtár neve).')

    parser.add_argument('--epochs', type=int, default=50,
                        help='A tanítási epoch-ok maximális száma (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch méret (default: 64)')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Adatbővítés kikapcsolása')
    return parser.parse_args()


def format_time(seconds):
    """Időt formáz óra:perc:másodperc formátumba"""
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

    print(f"\n--- ÚJ FUTTATÁS INDUL ---")
    print(f"Kísérlet neve: {RUN_NAME}")
    print(f"Minden eredmény ide lesz mentve: {RUN_RESULTS_DIR}")

    # --- 2. Adatok betöltése ---
    print("Adatok betöltése...")
    data = load_data()
    if data is None: return

    (X_train, y_train), (X_val, y_val, y_val_labels), X_test, num_classes, test_filenames = data
    input_shape = X_train.shape[1:]

    # --- 3. Modell építése ---
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
    model.summary()

    # --- 4. Callback-ek ---
    early_stopper = EarlyStopping(monitor='val_accuracy', patience=10,
                                  restore_best_weights=True, verbose=1)

    model_checkpoint_path = os.path.join(RUN_RESULTS_DIR, "best_model.keras")
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy',
                                       save_best_only=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=0.00001, verbose=1)

    callbacks_list = [early_stopper, model_checkpoint, reduce_lr]

    # --- 5. Tanítás időméréssel ---
    print(f"\n--- Tanítás indítása ({args.epochs} epoch) ---")
    training_start_time = time.time()

    if args.no_augmentation or args.model == 'mlp':
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                            validation_data=(X_val, y_val), callbacks=callbacks_list)
    else:
        print("Adatbővítés BEkapcsolva.")

        # Alapértelmezett beállítások (pl. a Hybrid modellhez)
        aug_config = {
            'rotation_range': 15,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'zoom_range': 0.05,
            'shear_range': 0.2
        }

        # Ha a 'size_expert' futtatást érzékeljük a névből, kapcsoljuk ki a zoomot!
        if "size_expert" in RUN_NAME:
            print(">> SPECIÁLIS MÓD: Size Expert (Zoom kikapcsolva!)")
            aug_config['zoom_range'] = 0.0
            aug_config['height_shift_range'] = 0.05  # Kevesebb függőleges mozgás is segít

        datagen = ImageDataGenerator(**aug_config)  # A ** kicsomagolja a szótárat
        datagen.fit(X_train)

        history = model.fit(datagen.flow(X_train, y_train, batch_size=args.batch_size),
                            epochs=args.epochs, validation_data=(X_val, y_val),
                            callbacks=callbacks_list)

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    actual_epochs = len(history.history['loss'])
    avg_time_per_epoch = total_training_time / actual_epochs

    print("\n--- Tanítás befejezve ---")
    print(f"Összes tanítási idő: {format_time(total_training_time)} ({total_training_time:.2f} másodperc)")
    print(f"Átlagos idő epochonként: {format_time(avg_time_per_epoch)} ({avg_time_per_epoch:.2f} másodperc)")
    print(f"Tényleges epoch-ok száma: {actual_epochs}")

    print(f"Legjobb modell betöltése innen: {model_checkpoint_path}")
    model = tf.keras.models.load_model(model_checkpoint_path)

    # --- 6. PREDIKCIÓ ---
    print("\n--- Végső predikció indítása a TESZT adatokon ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("Predikciók elkészültek.")

    # --- 7. Eredmények mentése ---
    print(f"Eredmények mentése a '{RUN_RESULTS_DIR}' mappába...")

    # Beadási fájl
    submission_df = pd.DataFrame({'class': y_pred, 'TestImage': test_filenames})
    submission_path = os.path.join(RUN_RESULTS_DIR, "submission.csv")
    submission_df.to_csv(submission_path, sep=';', index=False)
    print(f"Beadási fájl elmentve: {submission_path}")

    # Ábrák
    save_history_plot(history, os.path.join(RUN_RESULTS_DIR, "history.png"))
    save_misclassified_plot(model, X_val, y_val_labels,
                            os.path.join(RUN_RESULTS_DIR, "misclassified.png"))

    # Riport (validációs adatokon)
    report_path = os.path.join(RUN_RESULTS_DIR, "validation_report.txt")
    try:
        val_pred_probs = model.predict(X_val)
        val_pred = np.argmax(val_pred_probs, axis=1)
        report = classification_report(y_val_labels, val_pred)
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
        print(f"Validációs riport elmentve: {report_path}")
    except Exception as e:
        print(f"Validációs riport mentése sikertelen: {e}")

    print("\n--- Folyamat befejezve ---")


if __name__ == "__main__":
    main()