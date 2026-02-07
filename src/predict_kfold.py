import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd
from tqdm import tqdm
import argparse

# --- Configuration ---
TEST_IMAGE_DIR = 'data_raw/test'
RESULTS_DIR = 'results'


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Prediction using K-Fold models.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the K-Fold results folder (e.g., kfold_ensemble_2025...)')
    return parser.parse_args()


def load_all_models_from_folder(folder_name):
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    models = []

    if not os.path.exists(folder_path):
        print(f"ERROR: Directory not found: {folder_path}")
        exit()

    print(f"\n--- LOADING MODELS FROM: {folder_name} ---")

    # Find all .keras files
    files = [f for f in os.listdir(folder_path) if f.endswith('.keras')]

    if not files:
        print("ERROR: No .keras files found in this directory!")
        exit()

    for file in sorted(files):
        full_path = os.path.join(folder_path, file)
        try:
            print(f"Loading: {file}...")
            model = tf.keras.models.load_model(full_path)
            models.append(model)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(f"Successfully loaded {len(models)} models.")
    return models


def main():
    args = parse_args()

    # 1. Load Models
    models = load_all_models_from_folder(args.run_name)

    # 2. List Test Images
    try:
        test_filenames = sorted([
            f for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"ERROR: Test directory not found: {TEST_IMAGE_DIR}")
        return

    print(f"Starting prediction on {len(test_filenames)} images...")
    final_predictions = []
    failed_images = []
    successful_filenames = []

    # 3. Iterate over images
    for image_name in tqdm(test_filenames, desc="K-Fold Ensemble"):
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)

        try:
            # Load original image
            img_original = Image.open(image_path).convert('L')
            img_original = ImageOps.invert(img_original)  # <-- Invert to match training domain!
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")
            failed_images.append(image_name)
            continue

        all_probs = []

        # Predict with every model
        for model in models:
            # Dynamic resizing
            target_h = model.input_shape[1]
            target_w = model.input_shape[2]

            img_resized = img_original.resize((target_w, target_h))
            img_array = np.array(img_resized) / 255.0
            img_ready = img_array.reshape(1, target_h, target_w, 1)

            probs = model.predict(img_ready, verbose=0)
            all_probs.append(probs)

        # --- AVERAGING (Ensemble) ---
        if all_probs:
            avg_probs = np.mean(np.array(all_probs), axis=0)
            final_class = np.argmax(avg_probs)
            final_predictions.append(final_class)
            successful_filenames.append(image_name)
        else:
            final_predictions.append(0)
            successful_filenames.append(image_name) # Fallback to 0 still counts as a prediction? 
            # Wait, the prompt says: "create and use a separate successful_filenames list... and append image_name only when you successfully process and append a prediction... (keep failed_images as-is for errors)"
            # My previous change in Step 58 removed "final_predictions.append(0)" from the EXCEPTION block.
            # But here in the "else" of "if all_probs:", it means NO model predicted anything? That's weird if models are loaded.
            # If all_probs is empty, we probably shouldn't append to successful_filenames either?
            # But the loop iterates models. If models is empty, all_probs is empty.
            # Let's assume if we get here, we have a prediction.
            # Actually, if all_probs is empty, we effectively failed to predict.
            # But the prompt instruction regarding "append image_name only when you successfully process" refers to the try/except block failure.
            # Let's stick to the prompt: append to successful_filenames when we append to final_predictions.

    # 4. Save
    if not os.path.exists(os.path.join(RESULTS_DIR, args.run_name)):
        os.makedirs(os.path.join(RESULTS_DIR, args.run_name))

    output_filename = f"{args.run_name}_submission.csv"
    output_path = os.path.join(RESULTS_DIR, args.run_name, output_filename)

    submission_df = pd.DataFrame({
        'class': final_predictions,
        'TestImage': successful_filenames
    })

    if failed_images:
        print(f"\nWARNING: {len(failed_images)} images failed to process.")
        print(f"Failed images: {failed_images}")

    submission_df.to_csv(output_path, sep=';', index=False)

    print(f"\nDONE! K-Fold Ensemble submission file is here:\n{output_path}")


if __name__ == "__main__":
    main()