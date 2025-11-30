import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import argparse
import pandas as pd
import datetime

# Saját modulok
from utils import load_data_for_training_and_prediction as load_data
from visualize import save_history_plot, save_misclassified_plot, save_confusion_matrix_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Modell finomhangolása (Fine-Tuning).')

    # Melyik modellt akarjuk tovább tanítani? (A mappa neve a results-ban)
    parser.add_argument('--base_run', type=str, required=True,
                        help='A kiindulási modell mappájának neve (pl. v9_regularized...)')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Hány extra epoch-ot tanítsunk (default: 20)')

    return parser.parse_args()


def main():
    args = parse_args()

    BASE_RESULTS_DIR = 'results'
    BASE_MODEL_DIR = os.path.join(BASE_RESULTS_DIR, args.base_run)
    MODEL_PATH = os.path.join(BASE_MODEL_DIR, "best_model.keras")

    # Új mappa a finomhangolt eredményeknek
    NEW_RUN_NAME = f"{args.base_run}_FINETUNED"
    NEW_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, NEW_RUN_NAME)
    os.makedirs(NEW_RESULTS_DIR, exist_ok=True)

    print(f"\n--- FINE-TUNING INDÍTÁSA ---")
    print(f"Bázis modell: {MODEL_PATH}")
    print(f"Eredmények helye: {NEW_RESULTS_DIR}")

    # 1. Adatok betöltése
    print("Adatok betöltése...")
    data = load_data()  # Alapértelmezett (nem transfer)
    if data is None: return
    (X_train, y_train), (X_val, y_val, y_val_labels), X_test, num_classes, test_filenames = data

    # 2. A régi, legjobb modell betöltése
    try:
        print("Modell betöltése...")
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"HIBA: Nem sikerült betölteni a modellt: {e}")
        return

    # 3. ÚJRA-FORDÍTÁS (Re-compile) ALACSONY LEARNING RATE-TEL
    # Nagyon kicsi LR (pl. 0.00005), hogy csak finoman hangoljon.
    print("Modell újra-fordítása alacsony Learning Rate-tel (5e-5)...")

    # Megtartjuk a régi loss function-t (a label smoothing miatt), de új optimizert adunk neki
    new_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, clipnorm=1.0)

    # A loss function-t újra kell definiálni vagy kinyerni,
    # mert a load_model után néha elveszhet a custom beállítás.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(optimizer=new_optimizer, loss=loss_fn, metrics=['accuracy'])

    # 4. "Soft" Augmentáció beállítása
    # Sokkal gyengébb torzítások
    print("Enyhe (Soft) Augmentáció beállítása...")
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.0,
        shear_range=0.05
    )
    datagen.fit(X_train)

    # 5. Callback-ek
    early_stopper = EarlyStopping(monitor='val_accuracy', patience=5,
                                  restore_best_weights=True, verbose=1)

    model_checkpoint = ModelCheckpoint(os.path.join(NEW_RESULTS_DIR, "best_model.keras"),
                                       monitor='val_accuracy', save_best_only=True, verbose=1)

    # Itt is csökkenthetjük még tovább, ha kell
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    # 6. Finomhangolás futtatása
    print(f"\n--- Finomhangolás ({args.epochs} epoch) ---")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),  # Kisebb batch size a precizitásért
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopper, model_checkpoint, reduce_lr]
    )

    # 7. Eredmények mentése (Ugyanaz, mint a train.py-ban)
    print("Eredmények mentése...")

    # Submission
    print("Predikció teszt adatokra...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    sub_df = pd.DataFrame({'class': y_pred, 'TestImage': test_filenames})
    sub_df.to_csv(os.path.join(NEW_RESULTS_DIR, "submission.csv"), sep=';', index=False)

    # Ábrák
    save_history_plot(history, os.path.join(NEW_RESULTS_DIR, "history.png"))
    save_misclassified_plot(model, X_val, y_val_labels, os.path.join(NEW_RESULTS_DIR, "misclassified.png"))

    # Riport
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(y_val_labels, val_pred)
    with open(os.path.join(NEW_RESULTS_DIR, "validation_report.txt"), 'w') as f:
        f.write(report)

    # Konfúziós mátrix
    save_confusion_matrix_plot(y_val_labels, val_pred, os.path.join(NEW_RESULTS_DIR, "confusion_matrix.png"))

    print(f"\nKÉSZ! A finomhangolt modell itt található: {NEW_RESULTS_DIR}")


if __name__ == "__main__":
    main()