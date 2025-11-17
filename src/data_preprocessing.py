import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# --- 1. Konfiguráció ---
IMG_SIZE = 28
TRAIN_DATA_DIR = 'data_raw/train'
TEST_IMAGE_DIR = 'data_raw/test'  # Mappa, ahol a teszt képek "laposan" vannak
OUTPUT_DIR = 'data_processed'


# --- 2. Függvény a TANÍTÓ adatokhoz (MAPPÁKBÓL) ---
def process_train_images(data_dir):
    image_data_list = []
    label_list = []

    try:
        entries = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"HIBA: A '{data_dir}' mappa nem található!")
        return None, None, None

    sample_folders = [name for name in entries if os.path.isdir(os.path.join(data_dir, name))]

    if not sample_folders:
        print(f"HIBA: A '{data_dir}' mappában nem találhatók osztály-mappák!")
        return None, None, None

    label_map = {folder_name: i for i, folder_name in enumerate(sample_folders)}

    print(f"Tanító mappák feldolgozása a(z) '{data_dir}' könyvtárból...")
    print(f"Talált osztályok: {len(label_map)}")

    for folder_name in tqdm(sample_folders, desc="Tanító mappák"):
        folder_path = os.path.join(data_dir, folder_name)
        current_label = label_map[folder_name]

        for image_name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                pixel_array = np.array(img)
                flattened_array = pixel_array.flatten()

                image_data_list.append(flattened_array)
                label_list.append(current_label)
            except Exception as e:
                print(f"\nHiba a(z) {image_path} fájl feldolgozása közben: {e}")

    features_X = np.array(image_data_list)
    labels_y = np.array(label_list)

    return features_X, labels_y, label_map

# --- 3. Függvény a TESZT adatokhoz (LAPOS MAPPÁBÓL) ---
def process_test_images_no_labels(image_dir):
    """
    Feldolgozza a "lapos" teszt mappában lévő képeket.
    Címkéket nem keres, csak a képeket és a fájlneveiket menti.
    """
    image_data_list = []
    filename_list = []

    try:
        # Csak a képfájlokat szűrjük ki és rendezzük
        image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"HIBA: A '{image_dir}' teszt mappa nem található!")
        return None, None

    if not image_filenames:
        print(f"HIBA: Nem található képfájl a '{image_dir}' mappában!")
        return None, None

    print(f"Címke nélküli teszt képek feldolgozása a(z) '{image_dir}' mappából...")

    for image_name in tqdm(image_filenames, desc="Teszt képek"):
        image_path = os.path.join(image_dir, image_name)
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            pixel_array = np.array(img)
            flattened_array = pixel_array.flatten()

            image_data_list.append(flattened_array)
            filename_list.append(image_name)
        except Exception as e:
            print(f"\nHiba a(z) {image_path} fájl feldolgozása közben: {e}")

    features_X = np.array(image_data_list)
    filenames = np.array(filename_list)

    return features_X, filenames


# --- 4. Fő futtatható rész ---
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Tanító adatok feldolgozása ---
    print("--- Tanító adatok feldolgozása ---")
    X_train, y_train, label_map = process_train_images(TRAIN_DATA_DIR)

    if X_train is not None:
        np.save(os.path.join(OUTPUT_DIR, 'train_features.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), y_train)
        with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
            json.dump(label_map, f, indent=4)
        print(f"A tanító adatok elmentve: {X_train.shape}")
    else:
        print("HIBA: A tanító adatok feldolgozása sikertelen.")

    # --- Teszt adatok feldolgozása---
    print("\n--- Teszt adatok feldolgozása ---")
    X_test, test_filenames = process_test_images_no_labels(TEST_IMAGE_DIR)

    if X_test is not None:
        np.save(os.path.join(OUTPUT_DIR, 'test_features.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'test_filenames.npy'), test_filenames)
        print(f"A teszt képpontok elmentve: {X_test.shape}")
        print(f"A teszt fájlnevek elmentve: {test_filenames.shape}")
    else:
        print("HIBA: A teszt adatok feldolgozása sikertelen.")

    print("\nAdat-előkészítés befejezve!")