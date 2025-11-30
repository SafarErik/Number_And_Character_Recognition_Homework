import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse

from src.utils import IMG_SIZE

# --- Konfiguráció ---
TEST_IMAGE_DIR = 'data_raw/test'
RESULTS_DIR = 'results'


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble (több modell) jóslat készítése a TESZT adatokra.')
    parser.add_argument('--runs', nargs='+', required=True,
                        help='A futtatások nevei (mappák a results/ alatt).')
    parser.add_argument('--output_name', type=str, default='final_ensemble_submission',
                        help='A kimeneti fájl neve.')
    return parser.parse_args()


def load_models(run_names):
    models = []
    for run_name in run_names:
        model_path = os.path.join(RESULTS_DIR, run_name, 'best_model.keras')
        try:
            print(f"Modell betöltése: {run_name}...")
            model = tf.keras.models.load_model(model_path)
            models.append(model)
        except Exception as e:
            print(f"HIBA: Nem sikerült betölteni a modellt: {model_path}")
            print(e)
            exit()
    return models


def main():
    args = parse_args()

    # 1. Modellek betöltése
    models = load_models(args.runs)
    print(f"\nSikeresen betöltve {len(models)} modell.")

    # 2. Teszt fájlok listázása
    try:
        test_filenames = sorted([
            f for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"HIBA: Nem található a teszt mappa: {TEST_IMAGE_DIR}")
        exit()

    if not test_filenames:
        print("HIBA: Nincsenek képek a teszt mappában.")
        exit()

    print(f"Predikció indítása {len(test_filenames)} képen...")

    final_predictions = []

    # 3. Végigmegyünk a képeken
    for image_name in tqdm(test_filenames, desc="Ensemble Predikció"):
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)

        try:
            # Kép betöltése és előkészítése (Ugyanúgy, mint a tanításnál!)
            img = Image.open(image_path).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalizálás

            # (1, 32, 32, 1) formátum
            img_ready = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        except Exception as e:
            print(f"Hiba a kép feldolgozásakor ({image_name}): {e}")
            # Hiba esetén 0-t tippelünk (vagy bármit, hogy ne álljon meg a kód)
            final_predictions.append(0)
            continue

        # --- ENSEMBLE LOGIKA ---
        all_probs = []
        for model in models:
            probs = model.predict(img_ready, verbose=0)
            all_probs.append(probs)

        # NumPy tömbbé alakítás a súlyozáshoz
        stacked_probs = np.array(all_probs)  # Shape: (Modellek, 1, 62)
        # Az extra dimenziót (1) el kell tüntetni a shape-ből
        stacked_probs = np.squeeze(stacked_probs, axis=1)

        # --- SÚLYOZÁS ---
        # Ugyanazokat a súlyok, mint az analyze_ensemble.py-ban
        # Sorrend: [FineTuned, Shape, Structure]
        weights = [0.6, 0.2, 0.2]

        if len(models) == len(weights):
            avg_probs = np.average(stacked_probs, axis=0, weights=weights)
        else:
            # Ha véletlenül nem 3, sima átlagolás lesz.
            avg_probs = np.mean(stacked_probs, axis=0)

        final_class = np.argmax(avg_probs)
        final_predictions.append(final_class)

    # 4. Eredmény mentése
    output_path = os.path.join(RESULTS_DIR, f"{args.output_name}.csv")

    submission_df = pd.DataFrame({
        'class': final_predictions,
        'TestImage': test_filenames
    })

    submission_df.to_csv(output_path, sep=';', index=False)

    print(f"\nKimeneti fájl mentve: {output_path}")

if __name__ == "__main__":
    main()