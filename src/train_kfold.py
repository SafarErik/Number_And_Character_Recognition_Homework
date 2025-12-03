import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import argparse
import datetime
import gc

from utils import (
    IMG_SIZE,
    PROCESSED_DATA_DIR,
    morphological_augmentation
)

# --- MODELLEK IMPORTÁLÁSA ---
from models import (
    build_simple_cnn,
    build_advanced_cnn,
    build_keras_mlp,
    build_hybrid_cnn,
    build_pro_hybrid_cnn,
    build_regularized_hybrid_cnn,
    build_deep_hybrid_cnn
)

# --- MODELL VÁLASZTÓ SZÓTÁR ---
# Ez köti össze a szöveges nevet a konkrét függvénnyel
MODEL_BUILDERS = {
    'simple': build_simple_cnn,
    'advanced': build_advanced_cnn,
    'mlp': build_keras_mlp,
    'hybrid': build_hybrid_cnn,
    'pro_hybrid': build_pro_hybrid_cnn,
    'regularized': build_regularized_hybrid_cnn,
    'deep_hybrid': build_deep_hybrid_cnn
}

# GPU Fix
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser(description='Általános K-Fold Keresztvalidáció.')

    # --- ÚJ ARGUMENTUM: MODELL VÁLASZTÁS ---
    parser.add_argument('--model', type=str, default='regularized',
                        choices=list(MODEL_BUILDERS.keys()),  # Automatikusan felsorolja a lehetőségeket
                        help='Melyik modellt validáljuk? (default: regularized)')

    parser.add_argument('--folds', type=int, default=5, help='Foldok száma')
    parser.add_argument('--epochs', type=int, default=40, help='Epochok száma')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch méret')
    parser.add_argument('--run_name_prefix', type=str, default="kfold", help='Mentés előtagja')

    return parser.parse_args()


def main():
    args = parse_args()

    # A mappa nevébe beleírjuk a választott modellt is, hogy ne keveredjenek
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_name = f"{args.run_name_prefix}_{args.model}_{timestamp}"
    BASE_SAVE_DIR = os.path.join('results', full_run_name)

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    print(f"\n--- K-FOLD INDULÁSA: {args.model.upper()} MODELL ---")
    print(f"Képméret: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Foldok száma: {args.folds}")
    print(f"Mentés helye: {BASE_SAVE_DIR}")

    # 1. ADATOK
    print("Adatok betöltése...")
    try:
        X_raw = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'))
        y_raw = np.load(os.path.join(PROCESSED_DATA_DIR, 'train_labels.npy'))
    except Exception as e:
        print(f"Hiba: {e}")
        return

    X_full = X_raw / 255.0
    X_full = X_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    num_classes = np.max(y_raw) + 1

    # 2. CIKLUS
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    # Kiválasztjuk a megfelelő építő függvényt a szótárból
    model_builder_func = MODEL_BUILDERS[args.model]

    for train_index, val_index in skf.split(X_full, y_raw):
        print(f"\n" + "=" * 40)
        print(f"   FOLD {fold_no} / {args.folds} ({args.model})")
        print(f"=" * 40)

        X_train, X_val = X_full[train_index], X_full[val_index]
        y_train_raw, y_val_raw = y_raw[train_index], y_raw[val_index]

        y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes)
        y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes)

        # Augmentáció (Morfológiával)
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.05,
            preprocessing_function=morphological_augmentation
        )
        datagen.fit(X_train)

        # --- DINAMIKUS MODELL ÉPÍTÉS ---
        # Itt hívjuk meg a kiválasztott függvényt!
        model = model_builder_func(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=num_classes)

        checkpoint_path = os.path.join(BASE_SAVE_DIR, f"model_fold_{fold_no}.keras")

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=args.batch_size),
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Kiértékelés
        model.load_weights(checkpoint_path)
        scores = model.evaluate(X_val, y_val, verbose=0)
        acc = scores[1] * 100
        print(f"-> Fold {fold_no} pontossága: {acc:.2f}%")
        accuracies.append(acc)

        del model
        tf.keras.backend.clear_session()
        gc.collect()
        fold_no += 1

    print(f"\nÁTLAGOS PONTOSSÁG ({args.model}): {np.mean(accuracies):.2f}%")


if __name__ == "__main__":
    main()