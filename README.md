# Handwritten Character Recognition CNN

This project implements a Convolutional Neural Network (CNN) to recognize handwritten characters and digits. It features a modular Python package structure, multiple model architectures (Simple CNN, Advanced CNN, ResNet50 Transfer Learning, Hybrid CNN), and utilities for data preprocessing, training, evaluation, and hyperparameter tuning.

## Features

- **Multiple Architectures**: Includes Simple, Advanced, Hybrid, and ResNet-based models.
- **Data Augmentation**: Supports real-time data augmentation (rotation, shift, shear, zoom) and custom morphological augmentation (erosion/dilation).
- **K-Fold Cross-Validation**: Robust evaluation using Stratified K-Fold.
- **Hyperparameter Tuning**: Integration with Keras Tuner for optimizing model parameters.
- **Visualization**: Generates training history plots, confusion matrices, and misclassified image examples.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install the required Python packages**:
    ```bash
    pip install .
    # OR
    pip install tensorflow numpy pandas pillow tqdm scikit-learn matplotlib seaborn keras-tuner opencv-python-headless
    ```

## Usage

### 1. Data Preparation

1.  Create a `data_raw/` directory in the project root.
2.  Place your raw data into `data_raw/train` and `data_raw/test`.
    *   **Structure:**
        ```text
        data_raw/
        ├── train/
        │   ├── Sample001/ (e.g., images for class '0')
        │   ├── Sample002/ (e.g., images for class '1')
        │   └── ...
        └── test/
            ├── Test0001.png
            ├── ...
        ```
3.  Run the preprocessing script to generate optimized `.npy` files:
    ```bash
    python src/data_preprocessing.py
    ```
    This creates a `data_processed/` directory containing the processed datasets.

### 2. Training a Model

The main training script is `src/train.py`. You can specify the model architecture and other parameters.

**Example: Train the 'advanced' model:**
```bash
python src/train.py --model advanced --epochs 50 --run_name "advanced_run_v1"
```

**Example: Train without data augmentation:**
```bash
python src/train.py --model advanced --no_augmentation --run_name "advanced_no_aug"
```

**Available Arguments:**
*   `--model`: Model architecture (`simple`, `advanced`, `mlp`, `hybrid`, `pro_hybrid`, `regularized`, `deep_hybrid`, `resnet`).
*   `--run_name`: Unique name for the experiment (results will be saved in `results/<run_name>`).
*   `--epochs`: Number of training epochs (default: 50).
*   `--batch_size`: Batch size (default: 64).
*   `--no_augmentation`: Disable real-time data augmentation.

### 3. K-Fold Cross-Validation

To evaluate a model using K-Fold Cross-Validation:

```bash
python src/train_kfold.py --model regularized --folds 5 --epochs 40
```

### 4. Ensemble Prediction

After running K-Fold training, you can generate an ensemble prediction using all trained fold models:

```bash
python src/predict_kfold.py --run_name "kfold_regularized_2025..."
```

### 5. Hyperparameter Tuning

To optimize the hyperparameters of the Hybrid CNN model:

```bash
python src/tune_hybrid.py
```

### 6. Interactive Testing

You can test your models interactively by drawing on a canvas:

**Single Model Tester:**
```bash
python interactive_tester.py --run_name "advanced_run_v1"
```

**Ensemble Model Tester:**
```bash
python interactive_ensemble_tester.py --runs "model1_folder" "model2_folder"
```

## Results

All training results are saved in the `results/` directory, organized by run name. Each run directory contains:
*   `best_model.keras`: The saved model with the highest validation accuracy.
*   `history.png`: Training and validation accuracy/loss curves.
*   `confusion_matrix.png`: Confusion matrix of the validation set keys.
*   `misclassified.png`: Examples of misclassified images.
*   `submission.csv`: Predictions for the test set.
*   `validation_report.txt`: Detailed classification report.

## License

This project is open-source and available under the standard MIT license.