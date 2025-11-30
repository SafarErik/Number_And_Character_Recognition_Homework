import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Importáljuk az adatbetöltőt ÉS az IMG_SIZE-t a src-ből, hogy konzisztens legyen
from src.utils import load_data_for_training_and_prediction, IMG_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble hibaelemzés a validációs adatokon.')
    parser.add_argument('--runs', nargs='+', required=True,
                        help='A modellek mappanevei (pl. expert_size expert_shape hybrid_v6)')
    return parser.parse_args()


def load_models(run_names):
    models = []
    model_names = []
    for run_name in run_names:
        model_path = os.path.join('results', run_name, 'best_model.keras')
        try:
            print(f"Modell betöltése: {run_name}...")
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            model_names.append(run_name)
        except Exception as e:
            print(f"HIBA: Nem sikerült betölteni: {model_path}")
            print(e)
            exit()
    return models, model_names


def main():
    args = parse_args()

    # 1. Modellek betöltése
    models, model_names = load_models(args.runs)

    # 2. Validációs adatok betöltése
    print("Validációs adatok betöltése...")
    data = load_data_for_training_and_prediction()
    if data is None: return

    _, (X_val, _, y_val_true), _, num_classes, _ = data

    print(f"Validációs készlet mérete: {len(X_val)} kép")
    print(f"Képméret: {IMG_SIZE}x{IMG_SIZE}")

    # 3. Predikciók összegyűjtése
    print("Szakértők meghallgatása (Predikció)...")
    all_model_probs = []

    for i, model in enumerate(models):
        print(f"  -> {model_names[i]} gondolkodik...")
        try:
            probs = model.predict(X_val, verbose=0)
            all_model_probs.append(probs)
        except ValueError as e:
            print(f"HIBA a modell futtatásakor ({model_names[i]}): {e}")
            return

    # 4. Ensemble döntés (SÚLYOZOTT Átlagolás)
    # Először alakítsuk át a listát egy NumPy tömbbé
    stacked_probs = np.array(all_model_probs)

    # --- SÚLYOK BEÁLLÍTÁSA ---
    weights = [0.6, 0.2, 0.2]  # 3 modellhez

    print(f"Súlyozott átlagolás alkalmazása: {weights}")

    avg_probs = np.average(stacked_probs, axis=0, weights=weights)

    ensemble_predictions = np.argmax(avg_probs, axis=1)

    # ------------------------------------------------------------------

    # 5. Hibák keresése
    error_indices = np.where(ensemble_predictions != y_val_true)[0]

    print(f"\n--- EREDMÉNYEK ---")
    print(f"Összes validációs kép: {len(y_val_true)}")
    print(f"Ensemble tévedések száma: {len(error_indices)}")
    print(f"Ensemble Validációs Pontosság: {(1 - len(error_indices) / len(y_val_true)) * 100:.2f}%")

    if len(error_indices) == 0:
        print("Tökéletes! Nincs hiba.")
        return

    # 6. Vizualizáció (Max 15 hiba)
    num_to_show = min(15, len(error_indices))
    indices_to_show = np.random.choice(error_indices, num_to_show, replace=False)

    rows = int(np.ceil(num_to_show / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(20, 6 * rows))
    axes = axes.flatten()

    print(f"\n{num_to_show} db hiba részletes elemzése...")

    for i, idx in enumerate(indices_to_show):
        ax = axes[i]

        img = (X_val[idx].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        true_label = y_val_true[idx]
        ensemble_pred = ensemble_predictions[idx]

        title_text = f"Index: {idx} | VALÓS: {true_label} | ENSEMBLE: {ensemble_pred}\n"
        title_text += "-" * 30 + "\n"

        for m_idx, name in enumerate(model_names):
            p = all_model_probs[m_idx][idx]
            pred = np.argmax(p)
            conf = np.max(p) * 100
            mark = "✅" if pred == true_label else "❌"
            short_name = name
            title_text += f"{mark} {short_name}: {pred} ({conf:.1f}%)\n"

        ax.set_title(title_text, fontsize=10, loc='left', family='monospace')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()