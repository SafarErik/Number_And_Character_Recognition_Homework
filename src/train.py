import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import pandas as pd

# Saját modulok importálása
from utils import load_data_for_training_and_prediction as load_data
from models import build_simple_cnn, build_advanced_cnn, build_keras_mlp
from visualize import save_history_plot


# --- 1. Konfiguráció és Parancssori Argumentumok ---
def parse_args():
    parser = argparse.ArgumentParser(description='Karakterfelismerő modell tanítása.')
    parser.add_argument('--model', type=str, default='advanced',
                        choices=['simple', 'advanced', 'mlp'],
                        help='A használni kívánt modell típusa (default: advanced)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='A tanítási epoch-ok maximális száma (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch méret (default: 64)')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Adatbővítés kikapcsolása')
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_NAME = f"model_{args.model}"
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 2. Adatok betöltése ---
    print("Adatok betöltése...")
    data = load_data()
    if data is None:
        return

    (X_train, y_train), (X_val, y_val), X_test, num_classes, test_filenames = data

    input_shape = X_train.shape[1:]

    print(f"Tanító adatok: {X_train.shape}")
    print(f"Validációs adatok: {X_val.shape}")
    print(f"Predikciós (teszt) adatok: {X_test.shape}")
    print(f"Osztályok száma: {num_classes}")

    # --- 3. Modell kiválasztása és építése ---
    print(f"Modell építése: {args.model}")
    if args.model == 'simple':
        model = build_simple_cnn(input_shape, num_classes)
    elif args.model == 'advanced':
        model = build_advanced_cnn(input_shape, num_classes)
    elif args.model == 'mlp':
        model = build_keras_mlp(input_shape, num_classes)

    model.summary()

    # --- 4. Callback-ek ---
    early_stopper = EarlyStopping(monitor='val_accuracy', patience=10,
                                  restore_best_weights=True, verbose=1)

    model_checkpoint_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_best.keras")
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy',
                                       save_best_only=True, verbose=1)

    # --- 5. Tanítás ---
    print(f"\n--- Tanítás indítása ({args.epochs} epoch) ---")

    if args.no_augmentation or args.model == 'mlp':
        if args.model != 'mlp': print("Adatbővítés KIkapcsolva.")
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopper, model_checkpoint]
        )
    else:
        print("Adatbővítés BEkapcsolva.")
        datagen = ImageDataGenerator(
            rotation_range=10, width_shift_range=0.1,
            height_shift_range=0.1, zoom_range=0.1
        )
        datagen.fit(X_train)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=args.batch_size),
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopper, model_checkpoint]
        )

    print("--- Tanítás befejezve ---")

    print(f"Legjobb modell betöltése innen: {model_checkpoint_path}")
    model = tf.keras.models.load_model(model_checkpoint_path)

    # --- 6. PREDIKCIÓ a CÍMKE NÉLKÜLI TESZT adatokon ---
    print("\n--- Végső predikció indítása a TESZT adatokon ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Visszaalakítás one-hot-ból (pl. 0, 1, 19, stb.)
    print("Predikciók elkészültek.")

    # --- 7. Eredmények mentése ---
    print("Eredmények mentése a 'results/' mappába...")

    # --- BEADÁSI FÁJL LÉTREHOZÁSA ---
    submission_df = pd.DataFrame({
        'class': y_pred,
        'TestImage': test_filenames
    })
    submission_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_submission.csv")
    submission_df.to_csv(submission_path, sep=';', index=False)
    print(f"Beadási fájl (submission) elmentve: {submission_path}")

    # A tanítási görbét elmentjük.
    save_history_plot(history, os.path.join(RESULTS_DIR, f"{MODEL_NAME}_history.png"))

    print("\n--- Folyamat befejezve ---")


if __name__ == "__main__":
    main()