import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# --- 1. Konfiguráció ---
IMG_SIZE = 28
TRAIN_DATA_DIR = 'data_raw/train'
TEST_DATA_DIR = 'data_raw/test'
OUTPUT_DIR = 'data_processed'


# --- 2. Fő feldolgozó függvény ---
def process_images(data_dir):
    image_data_list = []
    label_list = []

    try:
        sample_folders = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"HIBA: A '{data_dir}' mappa nem található!")
        return None, None, None

    # Címke-szótár: pl: {'Sample001': 0, 'Sample002': 1, ...}
    label_map = {folder_name: i for i, folder_name in enumerate(sample_folders)}

    print(f"Mappák feldolgozása a(z) '{data_dir}' könyvtárból...")
    print(f"Talált osztályok: {len(label_map)}")

    for folder_name in tqdm(sample_folders, desc="Mappák feldolgozása"):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        current_label = label_map[folder_name]

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            try:
                img = Image.open(image_path).convert('L')  # Beolvasás és fekete-fehérré alakítás
                img = img.resize((IMG_SIZE, IMG_SIZE))  # Átméretezés
                pixel_array = np.array(img)
                flattened_array = pixel_array.flatten()  # Kilapítás (784,)

                image_data_list.append(flattened_array)
                label_list.append(current_label)
            except Exception as e:
                print(f"\nHiba a(z) {image_path} fájl feldolgozása közben: {e}")

    features_X = np.array(image_data_list)
    labels_y = np.array(label_list)

    return features_X, labels_y, label_map


# --- 3. Fő futtatható rész ---
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Tanító adatok feldolgozása ---
    print("--- Tanító adatok feldolgozása ---")
    X_train, y_train, label_map = process_images(TRAIN_DATA_DIR)

    if X_train is not None:
        np.save(os.path.join(OUTPUT_DIR, 'train_features.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), y_train)

        # Címke-szótár elmentése
        with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
            json.dump(label_map, f, indent=4)

        print(f"A tanító adatok elmentve: {X_train.shape}")
        print(f"A címke-szótár elmentve: {os.path.join(OUTPUT_DIR, 'label_map.json')}")

    # --- Teszt adatok feldolgozása ---
    print("\n--- Teszt adatok feldolgozása ---")
    X_test, y_test, _ = process_images(TEST_DATA_DIR)

    if X_test is not None:
        np.save(os.path.join(OUTPUT_DIR, 'test_features.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), y_test)
        print(f"A teszt adatok elmentve: {X_test.shape}")

    print("\nAdat-előkészítés befejezve!")