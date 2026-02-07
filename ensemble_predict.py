import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse

from src.utils import IMG_SIZE

# --- Configuration ---
TEST_IMAGE_DIR = 'data_raw/test'
RESULTS_DIR = 'results'


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble (multi-model) prediction on TEST data.')
    parser.add_argument('--runs', nargs='+', required=True,
                        help='Run names (folders under results/).')
    parser.add_argument('--output_name', type=str, default='final_ensemble_submission',
                        help='Output file name.')
    return parser.parse_args()


def load_models(run_names):
    models = []
    for run_name in run_names:
        model_path = os.path.join(RESULTS_DIR, run_name, 'best_model.keras')
        try:
            print(f"Loading model: {run_name}...")
            model = tf.keras.models.load_model(model_path)
            models.append(model)
        except Exception as e:
            print(f"ERROR: Failed to load model: {model_path}")
            print(e)
            exit()
    return models


def main():
    args = parse_args()

    # 1. Load Models
    models = load_models(args.runs)
    print(f"\nSuccessfully loaded {len(models)} models.")

    # 2. List Test Files
    try:
        test_filenames = sorted([
            f for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"ERROR: Test directory not found: {TEST_IMAGE_DIR}")
        exit()

    if not test_filenames:
        print("ERROR: No images found in the test directory.")
        exit()

    print(f"Starting prediction on {len(test_filenames)} images...")

    final_predictions = []

    # 3. Process Images
    for image_name in tqdm(test_filenames, desc="Ensemble Prediction"):
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)

        try:
            # Load and prepare image (Same as training!)
            img = Image.open(image_path).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize

            # Format: (1, 64, 64, 1) or whatever IMG_SIZE is
            img_ready = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        except Exception as e:
            print(f"Error processing image ({image_name}): {e}")
            # Predict 0 on error (or safe fallback)
            final_predictions.append(0)
            continue

        # --- ENSEMBLE LOGIC ---
        all_probs = []
        for model in models:
            # Adjust input shape if necessary (though IMG_SIZE should trigger resize match)
            # Check model input shape to be safe?
            # model_img_size = model.input_shape[1]
            # if model_img_size != IMG_SIZE: ... (Skipping for now assuming consistency)

            probs = model.predict(img_ready, verbose=0)
            all_probs.append(probs)

        # Convert to NumPy array for weighting
        stacked_probs = np.array(all_probs)  # Shape: (Models, 1, 62)
        # Remove extra dimension (1) from shape
        stacked_probs = np.squeeze(stacked_probs, axis=1)

        # --- WEIGHTING ---
        # Example weights: [FineTuned, Shape, Structure]
        # Modify this logic if dynamic weighting is needed
        weights = None
        if len(models) == 3:
             weights = [0.6, 0.2, 0.2]

        if weights and len(models) == len(weights):
            avg_probs = np.average(stacked_probs, axis=0, weights=weights)
        else:
            # Simple average if counts don't match
            avg_probs = np.mean(stacked_probs, axis=0)

        final_class = np.argmax(avg_probs)
        final_predictions.append(final_class)

    # 4. Save Results
    output_path = os.path.join(RESULTS_DIR, f"{args.output_name}.csv")

    submission_df = pd.DataFrame({
        'class': final_predictions,
        'TestImage': test_filenames
    })

    submission_df.to_csv(output_path, sep=';', index=False)

    print(f"\nOutput file saved: {output_path}")

if __name__ == "__main__":
    main()