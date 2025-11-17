# Number_And_Character_Recognition_Homework
This homework was made for a university course on Machine Learning and Deep Learning at BME.

## Data Setup

The training and testing data are not included in this repository due to their size.

1.  **Download the dataset:** You can download the dataset from [this Google Drive link](https://drive.google.com/drive/folders/19SiLQ1_Jx-NcvRF6eY4-kGv3-e-rweyO?usp=drive_link).
2.  **Unzip the files:** Unzip the downloaded file.
3.  **Place the data:** Place the `train1` and `test1` folders into the root directory of this project.

The final folder structure should look like this:
```
character-recognition/
├── data_raw/
│   ├── train/
│   │   ├── Sample001/
│   │   └── ...
│   └── test/
│       └── ...
├── data_processed/
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── test_features.npy
│   ├── test_labels.npy
├── results/
│   ├── simple_cnn_accuracy.png
│   ├── advanced_cnn_report.txt
│   └── advanced_cnn_model.h5
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   (A nyers képek átalakítója)
│   ├── utils.py                (Adatbetöltő és plot-oló segédfüggvények)
│   ├── models.py               (Itt definiálod az MLP, CNN, stb. modelleket)
│   └── train.py                (A fő szkript, amit futtatsz)
└── .gitignore
└── README.md
```
