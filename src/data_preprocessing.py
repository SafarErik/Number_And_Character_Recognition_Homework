import os
import numpy as np
from PIL import Image  # A képfeldolgozáshoz (Pillow)
from tqdm import tqdm  # A haladási sávhoz
import json

# --- 1. Konfiguráció ---
# Állítsd be a kívánt egységes képméretet
IMG_SIZE = 28

# Beviteli mappák (a nyers képekkel)
TRAIN_DATA_DIR = 'data_raw/train'
TEST_DATA_DIR = 'data_raw/test'

# Kimeneti mappa (a feldolgozott .npy fájlok helye)
OUTPUT_DIR = 'data_processed'


# --- 2. Fő feldolgozó függvény ---

def process_images(data_dir):
    """
    Bejárja a megadott könyvtárat, beolvassa a képeket, átalakítja őket,
    és visszaadja a feature (X) és label (y) tömböket, valamint a címke-szótárat.
    """

    image_data_list = []
    label_list = []

    # A mappák nevének sorba rendezése kulcsfontosságú!
    # Így biztosítjuk, hogy a 'Sample001' mindig a 0-s címkét kapja.
    try:
        sample_folders = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"HIBA: A '{data_dir}' mappa nem található!")
        return None, None, None

    # Létrehozzuk a címke-szótárat, pl: {'Sample001': 0, 'Sample002': 1, ...}
    label_map = {folder_name: i for i, folder_name in enumerate(sample_folders)}

    print(f"Mappák feldolgozása a(z) '{data_dir}' könyvtárból...")
    print(f"Talált címkék (osztályok): {len(label_map)}")

    # Végigmegyünk minden egyes 'SampleXXX' mappán
    for folder_name in tqdm(sample_folders, desc="Mappák feldolgozása"):
        folder_path = os.path.join(data_dir, folder_name)

        # Ha valami a mappában nem mappa, azt kihagyjuk
        if not os.path.isdir(folder_path):
            continue

        current_label = label_map[folder_name]

        # Végigmegyünk az összes képen az adott mappában
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            try:
                # 1. Kép beolvasása
                img = Image.open(image_path)

                # 2. Átalakítás fekete-fehérré ('L' mód)
                img = img.convert('L')

                # 3. Átméretezés az egységes méretre
                img = img.resize((IMG_SIZE, IMG_SIZE))

                # 4. Kép átalakítása NumPy tömbbé (0-255 pixelértékek)
                pixel_array = np.array(img)

                # 5. "Kilapítás" (Flatten) 2D-ből (28x28) 1D vektorrá (784)
                flattened_array = pixel_array.flatten()

                # Hozzáadás a listákhoz
                image_data_list.append(flattened_array)
                label_list.append(current_label)

            except Exception as e:
                # Hibakezelés, ha egy kép sérült
                print(f"\nHiba a(z) {image_path} fájl feldolgozása közben: {e}")

    # A listák átalakítása végső NumPy tömbökké
    features_X = np.array(image_data_list)
    labels_y = np.array(label_list)

    return features_X, labels_y, label_map


# --- 3. Fő futtatható rész ---

if __name__ == "__main__":

    # Biztosítjuk, hogy a kimeneti mappa létezik
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Tanító adatok feldolgozása ---
    print("--- Tanító adatok feldolgozása ---")
    X_train, y_train, label_map = process_images(TRAIN_DATA_DIR)

    if X_train is not None:
        print(f"\nTanító adatok formája (X): {X_train.shape}")  # Pl. (50000, 784)
        print(f"Tanító címkék formája (y): {y_train.shape}")  # Pl. (50000,)

        # Fájlok mentése
        np.save(os.path.join(OUTPUT_DIR, 'train_features.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), y_train)

        # A címke-szótár elmentése (nagyon fontos a kiértékeléshez!)
        with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
            json.dump(label_map, f, indent=4)

        print(f"A tanító adatok elmentve a '{OUTPUT_DIR}' mappába.")
        print(f"A címke-szótár elmentve: {os.path.join(OUTPUT_DIR, 'label_map.json')}")

    # --- Teszt adatok feldolgozása ---
    print("\n--- Teszt adatok feldolgozása ---")
    # A teszt adatok feldolgozásánál ugyanazt a logikát használjuk
    X_test, y_test, _ = process_images(TEST_DATA_DIR)

    if X_test is not None:
        print(f"\nTeszt adatok formája (X): {X_test.shape}")
        print(f"Teszt címkék formája (y): {y_test.shape}")

        # Fájlok mentése
        np.save(os.path.join(OUTPUT_DIR, 'test_features.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), y_test)

        print(f"A teszt adatok elmentve a '{OUTPUT_DIR}' mappába.")

    print("\nAdat-előkészítés befejezve!")