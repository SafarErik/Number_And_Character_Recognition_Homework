import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# --- 1. Configuration ---
IMG_SIZE = 64
TRAIN_DATA_DIR = 'data_raw/train'
TEST_IMAGE_DIR = 'data_raw/test'
OUTPUT_DIR = 'data_processed'
VISUALIZATION_DIR = 'visualization'


# --- 2. Function for TRAINING data (FROM FOLDERS) ---
def process_train_images(data_dir):
    """
    Reads training images from folders where each folder name contains the class ID.
    Images are converted to grayscale, inverted, and resized to IMG_SIZE x IMG_SIZE.
    """
    image_data_list = []
    label_list = []

    try:
        entries = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"ERROR: The '{data_dir}' directory was not found!")
        return None, None

    sample_folders = [name for name in entries if os.path.isdir(os.path.join(data_dir, name))]

    if not sample_folders:
        print(f"ERROR: No class folders found in '{data_dir}'!")
        return None, None

    print(f"Processing training folders from '{data_dir}'...")
    print(f"Found Sample folders: {len(sample_folders)}")

    for folder_name in tqdm(sample_folders, desc="Training folders"):
        folder_path = os.path.join(data_dir, folder_name)

        # --- Extract label from folder name ---
        try:
            # Expected format: Sample001, Sample019 -> Class 1, 19
            class_id_str = folder_name.replace('Sample', '').lstrip('0')
            if not class_id_str:
                class_id_str = '0'
            current_label = int(class_id_str)
        except ValueError:
            print(f"\nWARNING: Folder '{folder_name}' does not match 'SampleXXX' format. Skipping.")
            continue

        for image_name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_name)
            try:
                img = Image.open(image_path).convert('L')

                # --- INVERSION (WHITE BACKGROUND -> BLACK BACKGROUND) ---
                # This helps the model focus on the strokes
                img = ImageOps.invert(img)
                # --------------------------------------------------

                img = img.resize((IMG_SIZE, IMG_SIZE))
                pixel_array = np.array(img)
                flattened_array = pixel_array.flatten()

                image_data_list.append(flattened_array)
                label_list.append(current_label)
            except Exception as e:
                print(f"\nError processing file {image_path}: {e}")

    if not image_data_list:
        print("ERROR: No valid images found in the training directories.")
        return None, None

    features_X = np.array(image_data_list)
    labels_y = np.array(label_list)

    return features_X, labels_y


# --- 3. Function for UNLABELED TEST data ---
def process_test_images_no_labels(image_dir):
    """
    Reads test images from a directory.
    Images are converted to grayscale, inverted, and resized.
    Returns features and filenames.
    """
    image_data_list = []
    filename_list = []
    try:
        image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    except FileNotFoundError:
        print(f"ERROR: The '{image_dir}' test directory was not found!")
        return None, None
    if not image_filenames:
        print(f"ERROR: No image files found in '{image_dir}'!")
        return None, None
    print(f"Processing unlabeled test images from '{image_dir}'...")

    for image_name in tqdm(image_filenames, desc="Test images"):
        image_path = os.path.join(image_dir, image_name)
        try:
            img = Image.open(image_path).convert('L')

            # --- INVERSION IS MANDATORY HERE TOO! ---
            img = ImageOps.invert(img)
            # ----------------------------------------

            img = img.resize((IMG_SIZE, IMG_SIZE))
            pixel_array = np.array(img)
            flattened_array = pixel_array.flatten()
            image_data_list.append(flattened_array)
            filename_list.append(image_name)
        except Exception as e:
            print(f"\nError processing file {image_path}: {e}")

    features_X = np.array(image_data_list)
    filenames = np.array(filename_list)
    return features_X, filenames


# --- 4. Create Visualization ---
def create_visualization_sample(train_dir, output_vis_dir, target_folder='Sample001'):
    """
    Creates a sample visualization of the preprocessing steps (Original -> Inverted & Resized).
    """
    sample_folder = os.path.join(train_dir, target_folder)
    if not os.path.isdir(sample_folder):
        print(f"WARNING: Folder '{target_folder}' not found, skipping visualization.")
        return
    image_files = sorted([
        f for f in os.listdir(sample_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_files:
        print(f"WARNING: No images found in '{target_folder}', skipping visualization.")
        return

    first_image_name = image_files[0]
    img_path = os.path.join(sample_folder, first_image_name)

    os.makedirs(output_vis_dir, exist_ok=True)
    # Clean up previous visualizations
    # Clean up previous visualizations
    for f in os.listdir(output_vis_dir):
        file_path = os.path.join(output_vis_dir, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            # If you want to handle directories, use shutil.rmtree here
        except OSError as e:
            print(f"Error removing {file_path}: {e}")

    try:
        # 1. Original Image (White background)
        original_img = Image.open(img_path)
        original_save_path = os.path.join(output_vis_dir, 'original.png')
        original_img.save(original_save_path)

        # 2. Processed Image (Inverted, Black background, 64x64)
        processed_img = original_img.convert('L')
        processed_img = ImageOps.invert(processed_img)  # <-- Invert for visualization too!
        processed_img = processed_img.resize((IMG_SIZE, IMG_SIZE))

        arr = np.array(processed_img).astype(np.float32) / 255.0
        arr_to_save = (arr * 255).astype(np.uint8)
        processed_save_path = os.path.join(output_vis_dir, 'processed_inverted_64x64.png')
        Image.fromarray(arr_to_save).save(processed_save_path)

        print(f"Visualization saved:")
        print(f" - Original: {original_save_path}")
        print(f" - Processed (As seen by model): {processed_save_path}")

    except Exception as e:
        print(f"ERROR creating visualization: {e}")


# --- 5. Main Execution ---
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Process Training Data ---
    print("--- Processing Training Data ---")
    X_train, y_train = process_train_images(TRAIN_DATA_DIR)

    if X_train is not None:
        np.save(os.path.join(OUTPUT_DIR, 'train_features.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), y_train)
        print(f"Training data saved: {X_train.shape}")
        print(f"Class IDs created (min/max): {np.min(y_train)} / {np.max(y_train)}")
        print(f"Number of unique classes: {len(np.unique(y_train))}")
    else:
        print("ERROR: Processing training data failed.")

    # --- Process Test Data ---
    print("\n--- Processing Test Data ---")
    X_test, test_filenames = process_test_images_no_labels(TEST_IMAGE_DIR)
    if X_test is not None:
        np.save(os.path.join(OUTPUT_DIR, 'test_features.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'test_filenames.npy'), test_filenames)
        print(f"Test pixel data saved: {X_test.shape}")
    else:
        print("ERROR: Processing test data failed.")

    # --- Create Visualization (Sample001 first image) ---
    print("\n--- Creating Visualization ---")
    create_visualization_sample(TRAIN_DATA_DIR, VISUALIZATION_DIR, target_folder='Sample001')

    print("\nData preprocessing complete! Please check the 'visualization' folder.")