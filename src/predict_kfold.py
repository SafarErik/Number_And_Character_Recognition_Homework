import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse

# --- Konfiguráció ---
TEST_IMAGE_DIR = 'data_raw/test'
RESULTS_DIR = 'results'


def parse_args():
    parser = argparse.ArgumentParser(description='K-Fold modellek közös jóslása.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='A K-Fold eredmény mappájának neve (pl. kfold_ensemble_2025...)')
    return parser.parse_args()


def load_all_models_from_folder(folder_name):
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    models = []

    if not os.path.exists(folder_path):
        print(f"HIBA: Nem található a mappa: {folder_path}")
        exit()

    print(f"\n--- MODELLEK BETÖLTÉSE INNEN: {folder_name} ---")

    # Megkeressük az összes .keras fájlt a mappában
    files = [f for f in os.listdir(folder_path) if f.endswith('.keras')]

    if not files:
        print("HIBA: Nem találtam .keras fájlokat ebben a mappában!")
        exit()

    for file in sorted(files):
        full_path = os.path.join(folder_path, file)
        try:
            print(f"Betöltés: {file}...")
            model = tf.keras.models.load_model(full_path)
            models.append(model)
        except Exception as e:
            print(f"Hiba a {file} betöltésekor: {e}")

    print(f"Sikeresen betöltve {len(models)} modell.")
    return models


def main():
    args = parse_args()

    # 1. Modellek betöltése
    models = load_all_models_from_folder(args.run_name)

    # 2. Teszt képek listázása
    try:
        test_filenames = sorted([
            f for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"HIBA: Nem található a teszt mappa: {TEST_IMAGE_DIR}")
        return

    print(f"Predikció indítása {len(test_filenames)} képen...")
    final_predictions = []

    # 3. Ciklus a képeken
    for image_name in tqdm(test_filenames, desc="K-Fold Ensemble"):
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)

        try:
            # Betöltjük az eredeti képet
            img_original = Image.open(image_path).convert('L')
        except Exception as e:
            final_predictions.append(0)
            continue

        all_probs = []

        # Minden modellel jósolunk
        for model in models:
            # Dinamikus méretezés (ha esetleg 32-es vagy 64-es lenne vegyesen)
            target_h = model.input_shape[1]
            target_w = model.input_shape[2]

            img_resized = img_original.resize((target_w, target_h))
            img_array = np.array(img_resized) / 255.0
            img_ready = img_array.reshape(1, target_h, target_w, 1)

            probs = model.predict(img_ready, verbose=0)
            all_probs.append(probs)

        # --- ÁTLAGOLÁS (Ensemble) ---
        # Itt sima átlagot használunk, mert a K-Fold modellek egyenrangúak
        avg_probs = np.mean(np.array(all_probs), axis=0)

        final_class = np.argmax(avg_probs)
        final_predictions.append(final_class)

    # 4. Mentés
    output_filename = f"{args.run_name}_submission.csv"
    output_path = os.path.join(RESULTS_DIR, args.run_name, output_filename)

    submission_df = pd.DataFrame({
        'class': final_predictions,
        'TestImage': test_filenames
    })

    submission_df.to_csv(output_path, sep=';', index=False)

    print(f"\nKÉSZ! A K-Fold Ensemble beadandó fájl itt van:\n{output_path}")


if __name__ == "__main__":
    main()