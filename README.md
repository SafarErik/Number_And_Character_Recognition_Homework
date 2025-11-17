# Karakterfelismerő CNN Projekt

Ez a projekt egy Konvolúciós Neurális Hálót (CNN) tanít be, ami képes felismerni kézzel írott számokat és betűket. A projekt professzionális, skálázható Python csomagstruktúrát használ.

## 🚀 Futtatás

### 1. Telepítés

A projekt futtatásához szükséges Python könyvtárak:
```bash
pip install tensorflow numpy pillow tqdm scikit-learn matplotlib seaborn
```
### 2. Adatok Előkészítése

1.  Hozd létre a `data_raw/` mappát a projekt gyökerében.
2.  Töltsd le a nyers adatokat (pl. a Google Drive linkedről) és másold őket a `data_raw/train` és `data_raw/test` mappákba.
    A várt struktúra:
    ```
    data_raw/
    ├── train/
    │   ├── Sample001/ (pl. '0' képei)
    │   ├── Sample002/ (pl. '1' képei)
    │   └── ...
    └── test/
        ├── Sample001/
        └── ...
    ```

3.  Futtasd az adat-előkészítő szkriptet. **Ezt csak egyszer kell megtenni.**
    ```bash
    python src/data_preprocessing.py
    ```
    Ez létrehozza a `data_processed/` mappát a tiszta `.npy` fájlokkal.

### 3. Modell Tanítása

A fő tanító szkript a `src/train.py`. Parancssori argumentumokkal vezérelheted:

**Alapértelmezett futtatás (fejlett CNN, 50 epoch, adatbővítéssel):**
```bash
python src/train.py
```

**Kísérletezés (pl. 'simple' modell, 20 epoch, adatbővítés nélkül):**
```bash
python src/train.py --model simple --epochs 20 --no_augmentation
```

**Elérhető argumentumok:**
* `--model`: Melyik modellt futtassa (`simple`, `advanced`, `mlp`).
* `--epochs`: Epoch-ok maximális száma.
* `--batch_size`: Batch méret.
* `--no_augmentation`: Kikapcsolja a valós idejű adatbővítést.

### 4. Eredmények

A tanítás végén az összes kimenet (modell, ábrák, riport) a `results/` mappába kerül, a modell neve alapján elnevezve (pl. `results/model_advanced_best.h5`).