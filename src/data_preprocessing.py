import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# --- 1. Konfiguráció ---
IMG_SIZE = 28
TRAIN_DATA_DIR = 'data_raw/train'
TEST_IMAGE_DIR = 'data_raw/test'
OUTPUT_DIR = 'data_processed'


# --- 2. Függvény a TANÍTÓ adatokhoz (MAPPÁKBÓL) ---
def process_train_images(data_dir):
    image_data_list = []
    label_list = []

    try:
        entries = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"HIBA: A '{data_dir}' mappa nem található!")
        return None, None

    sample_folders = [name for name in entries if os.path.isdir(os.path.join(data_dir, name))]

    if not sample_folders:
        print(f"HIBA: A '{data_dir}' mappában nem találhatók osztály-mappák!")
        return None, None

    print(f"Tanító mappák feldolgozása a(z) '{data_dir}' könyvtárból...")
    print(f"Talált Sample mappák: {len(sample_folders)}")

    for folder_name in tqdm(sample_folders, desc="Tanító mappák"):
        folder_path = os.path.join(data_dir, folder_name)

        # --- Címke kinyerése a mappanévből ---
        try:
            # Pl. 'Sample019' -> '019' -> 19
            # Pl. 'Sample045' -> '045' -> 45
            class_id_str = folder_name.replace('Sample', '').lstrip('0')
            if not class_id_str: class_id_str = '0'  # Pl. 'Sample000' esete

            current_label = int(class_id_str)

        except ValueError:
            print(f"\nFIGYELEM: A '{folder_name}' mappa neve nem 'SampleXXX' formátumú. Kihagyom.")
            continue

        for image_name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                pixel_array = np.array(img)
                flattened_array = pixel_array.flatten()

                image_data_list.append(flattened_array)
                label_list.append(current_label)  # pl. 45-ös címkét adjuk hozzá
            except Exception as e:
                print(f"\nHiba a(z) {image_path} fájl feldolgozása közben: {e}")

    features_X = np.array(image_data_list)
    labels_y = np.array(label_list)

    return features_X, labels_y


# --- 3. Függvény a CÍMKE NÉLKÜLI TESZT adatokhoz ---
def process_test_images_no_labels(image_dir):
    image_data_list = []
    filename_list = []
    try:
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
    X_train, y_train = process_train_images(TRAIN_DATA_DIR)

    if X_train is not None:
        np.save(os.path.join(OUTPUT_DIR, 'train_features.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), y_train)
        print(f"A tanító adatok elmentve: {X_train.shape}")
        print(f"Létrehozott osztály-azonosítók (min/max): {np.min(y_train)} / {np.max(y_train)}")
        print(f"Egyedi osztályok száma: {len(np.unique(y_train))}")
    else:
        print("HIBA: A tanító adatok feldolgozása sikertelen.")

    # --- Teszt adatok feldolgozása ---
    print("\n--- Teszt adatok feldolgozása ---")
    X_test, test_filenames = process_test_images_no_labels(TEST_IMAGE_DIR)
    if X_test is not None:
        np.save(os.path.join(OUTPUT_DIR, 'test_features.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'test_filenames.npy'), test_filenames)
        print(f"A teszt képpontok elmentve: {X_test.shape}")
    else:
        print("HIBA: A teszt adatok feldolgozása sikertelen.")

    print("\nAdat-előkészítés befejezve!")