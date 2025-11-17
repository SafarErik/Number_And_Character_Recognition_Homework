# Karakterfelismerő CNN Projekt

Ez a projekt egy Konvolúciós Neurális Hálót (CNN) tanít be, ami képes felismerni kézzel írott számokat és betűket. A projekt professzionális, skálázható Python csomagstruktúrát használ a kísérletek tiszta és átlátható követéséhez.

## Használat

### 1\. Telepítés

A projekt futtatásához szükséges Python könyvtárak:

```bash
pip install tensorflow numpy pillow tqdm scikit-learn matplotlib seaborn pandas
```

*(Érdemes ezeket egy `requirements.txt` fájlba is beírni.)*

-----

### 2\. Adatok Előkészítése

1.  Hozd létre a `data_raw/` mappát a projekt gyökerében.

2.  Töltsd le a nyers adatokat (pl. a Google Drive linkedről) és másold őket a `data_raw/train` és `data_raw/test` mappákba.

    **A várt struktúra:**

    ```
    data_raw/
    ├── train/
    │   ├── Sample001/ (pl. '1'-es osztály képei)
    │   ├── Sample019/ (pl. '19'-es osztály képei)
    │   └── ...
    └── test/
        ├── Test0001.png
        ├── Test0002.png
        └── ...
    ```

3.  Futtasd az adat-előkészítő szkriptet. **Ezt csak egyszer kell megtenni.**

    ```bash
    python src/data_preprocessing.py
    ```

    Ez létrehozza a `data_processed/` mappát a tiszta `.npy` fájlokkal, amiket a modell már fel tud dolgozni.

-----

### 3\. Modell Tanítása

A fő tanító szkript a `src/train.py`. Egy `--run_name` argumentummal vezérelheted, ami egy egyedi almappát hoz létre az eredményeknek a `results/` mappán belül.

**Egy kísérlet futtatása "advanced\_v1" néven:**

```bash
python src/train.py --model advanced --epochs 50 --run_name "advanced_v1"
```

**Kísérletezés adatbővítés nélkül:**

```bash
python src/train.py --model advanced --no_augmentation --run_name "advanced_no_aug"
```

**A "simple" modell futtatása:**

```bash
python src/train.py --model simple --run_name "simple_model_v1"
```

**Ha nem adsz meg `--run_name`-et, automatikusan generál egyet** (pl. `advanced_20251117_220000`).

#### Elérhető argumentumok:

  * `--model`: Melyik modellt futtassa (`simple`, `advanced`, `mlp`).
  * `--run_name`: A futtatás egyedi neve. **Ez lesz a mappa neve a `results`-ben.**
  * `--epochs`: Epoch-ok maximális száma.
  * `--batch_size`: Batch méret.
  * `--no_augmentation`: Kikapcsolja a valós idejű adatbővítést.

-----

### 4\. Eredmények Kiértékelése

A tanítás végén az összes kimenet a `results/RUN_NAME` mappába kerül (pl. `results/advanced_v1/`).

**A mappa tartalma:**

  * `best_model.keras`: A legjobb validációs pontosságot elért, betanított modell.
  * `history.png`: A tanítási és validációs pontosság/hiba görbéi.
  * `misclassified.png`: Példák a validációs adathalmazból, amiket a modell elrontott.
  * `submission.csv`: A végső, beadandó fájl a `test` adatokra adott jóslatokkal.
  * `validation_report.txt`: Egy részletes kiértékelés (precision, recall) a validációs adatokon.