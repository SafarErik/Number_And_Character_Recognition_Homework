import os
import numpy as np
from PIL import Image
import cv2  # OpenCV szükséges a paddingoláshoz (pip install opencv-python)
from tqdm import tqdm

# --- 1. Konfiguráció ---
IMG_SIZE = 32
TRAIN_DATA_DIR = 'data_raw/train'
TEST_IMAGE_DIR = 'data_raw/test'
OUTPUT_DIR = 'data_processed'
VISUALIZATION_DIR = 'visualization'


# --- 2. Képarány-megőrző átméretezés paddingolással ---
def preprocess_image_aspect_ratio(image_path, target_size=IMG_SIZE):
    """
    Képarány-megőrző átméretezés (paddinggal).
    Ez kritikus a '0' vs 'o' és 'w' vs 'W' megkülönböztetéséhez.
    Nem nyújtja meg a karaktert, hanem fekete keretet tesz köré.
    """
    try:
        # Kép betöltése szürkeárnyalatosként
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Ha a kép világos (fehér) hátterű, invertáljuk (fekete háttér, fehér betű kell)
        # Ez egyszerű heurisztika: ha az átlag pixelérték magas, akkor valszeg fehér a háttér.
        if np.mean(img) > 127:
            img = 255 - img

        # Eredeti méretek
        old_size = img.shape[:2] # (height, width)

        # Kiszámoljuk az arányt, hogy beleférjen a dobozba
        ratio = float(target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size]) # (new_height, new_width)

        # Átméretezés (INTER_AREA jobb kicsinyítéshez)
        img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

        # Kiszámoljuk mennyi keret kell
        delta_w = target_size - new_size[1]
        delta_h = target_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Fekete keret hozzáadása (Border)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        return new_img

    except Exception as e:
        print(f"\nHiba a kép feldolgozásakor ({image_path}): {e}")
        return None


# --- 3. Függvény a TANÍTÓ adatokhoz (MAPPÁKBÓL) ---
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
            class_id_str = folder_name.replace('Sample', '').lstrip('0')
            if not class_id_str:
                class_id_str = '0'
            current_label = int(class_id_str)
        except ValueError:
            print(f"\nFIGYELEM: A '{folder_name}' mappa neve nem 'SampleXXX' formátumú. Kihagyom.")
            continue

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)

            # Új paddingos feldolgozás
            processed_img = preprocess_image_aspect_ratio(image_path, IMG_SIZE)

            if processed_img is not None:
                image_data_list.append(processed_img)
                label_list.append(current_label)

    if not image_data_list:
        print("Nem sikerült képeket betölteni.")
        return None, None

    # 4D tömb létrehozása (samples, height, width, channels) - CNN-ekhez
    features_X = np.array(image_data_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels_y = np.array(label_list)

    return features_X, labels_y


# --- 4. Függvény a CÍMKE NÉLKÜLI TESZT adatokhoz ---
def process_test_images_no_labels(image_dir):
    image_data_list = []
    filename_list = []
    try:
        image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
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

        # Új paddingos feldolgozás
        processed_img = preprocess_image_aspect_ratio(image_path, IMG_SIZE)

        if processed_img is not None:
            image_data_list.append(processed_img)
            filename_list.append(image_name)

    if not image_data_list:
        print("Nem sikerült képeket betölteni.")
        return None, None

    # 4D tömb létrehozása (samples, height, width, channels) - CNN-ekhez
    features_X = np.array(image_data_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    filenames = np.array(filename_list)
    return features_X, filenames


# --- 5. Vizualizáció létrehozása ---
def create_visualization_sample(train_dir, output_vis_dir, target_folder='Sample001'):
    sample_folder = os.path.join(train_dir, target_folder)
    if not os.path.isdir(sample_folder):
        print(f"FIGYELEM: A '{target_folder}' mappa nem található, vizualizáció kihagyva.")
        return
    image_files = sorted([
        f for f in os.listdir(sample_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    if not image_files:
        print(f"FIGYELEM: Nincs képfájl a '{target_folder}' mappában, vizualizáció kihagyva.")
        return

    first_image_name = image_files[0]
    img_path = os.path.join(sample_folder, first_image_name)

    os.makedirs(output_vis_dir, exist_ok=True)
    # Tisztítás: mindig csak 2 fájl legyen
    for f in os.listdir(output_vis_dir):
        try:
            os.remove(os.path.join(output_vis_dir, f))
        except Exception:
            pass

    try:
        # Eredeti kép mentése (PIL-lel)
        original_img = Image.open(img_path)
        original_save_path = os.path.join(output_vis_dir, 'original.png')
        original_img.save(original_save_path)

        # Feldolgozott kép mentése (új módszerrel)
        processed_img = preprocess_image_aspect_ratio(img_path, IMG_SIZE)
        if processed_img is not None:
            processed_save_path = os.path.join(output_vis_dir, 'processed_32x32.png')
            cv2.imwrite(processed_save_path, processed_img)
            print(f"Vizualizáció mentve: {original_save_path}, {processed_save_path}")
        else:
            print("HIBA: Nem sikerült feldolgozni a képet a vizualizációhoz")
    except Exception as e:
        print(f"HIBA a vizualizáció létrehozásakor: {e}")


# --- 5. Fő futtatható rész ---
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

    # --- Vizualizáció (Sample001 első kép) ---
    print("\n--- Vizualizáció létrehozása ---")
    create_visualization_sample(TRAIN_DATA_DIR, VISUALIZATION_DIR, target_folder='Sample001')

    print("\nAdat-előkészítés befejezve!")