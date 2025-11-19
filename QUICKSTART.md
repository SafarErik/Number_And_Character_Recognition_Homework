# Gyors Példák - Optimized Model Használata

## 🚀 Gyors Start

### 1. Optimalizált modell tanítása (ajánlott)
```bash
cd "C:\_Tomi\_BME\_5.felev\Adatelemzés\HF\Program"
python src/train.py --model optimized --epochs 60 --run_name "opt_kiserlet_01"
```

### 2. Gyors teszt (5 epoch)
```bash
python src/train.py --model optimized --epochs 5 --run_name "gyors_teszt"
```

### 3. Nagy batch méret (GPU-val)
```bash
python src/train.py --model optimized --epochs 60 --batch_size 128 --run_name "opt_big_batch"
```

---

## 📊 Különbség az új és régi kód között

### Amit az ÚJ kód ad:

✅ **Focal Loss** - Automatikusan koncentrál a nehéz karakterekre (o/0, w/W)
✅ **Speciális Augmentáció** - Zoom TILTVA, hogy megőrizze a méretkülönbségeket
✅ **Megnövelt Türelem** - 12 epoch patience (vs 10) a Focal Loss hullámzása miatt
✅ **GPU Optimalizálás** - Automatikus memory growth
✅ **Részletes Statisztikák** - Tanítási idő, epoch átlag, stb.

### Főbb változások:

| Funkció | Régi | Új |
|---------|------|-----|
| Augmentáció zoom | 0.05 | **0.0** (TILTVA) |
| Rotation | 15° | 8° (finomabb) |
| Shift | 0.15 | 0.08 (finomabb) |
| Shear | 0.2 | 0.05 (minimális) |
| EarlyStopping patience | 10 | 12 (optimized-nél) |
| ReduceLR patience | 3 | 4 (optimized-nél) |

---

## 🎯 Mire jó az Optimized Model?

### Erősségek:
- **'o' vs '0'** - Focal Loss jobban megkülönbözteti
- **'w' vs 'W'** - Méret megőrzése (zoom tiltva)
- **'I' vs 'l' vs '1'** - Térbeli információ megtartása (Flatten)
- **Gyors konvergencia** - BatchNormalization minden rétegnél
- **Finom gradiens** - Swish aktiváció

### Gyengeségek:
- Kicsit lassabb konvergencia kezdetben (Focal Loss)
- Több türelem kell (12-15 epoch)
- Erősebb GPU ajánlott (2.7M paraméter)

---

## 📁 Eredmények Helye

Minden futtatás után:
```
results/
  └── opt_kiserlet_01/
      ├── best_model.keras         # Legjobb modell
      ├── submission.csv            # Beadási fájl
      ├── history.png              # Görbék
      ├── misclassified.png        # Hibák
      └── validation_report.txt    # Teljes riport
```

---

## 🔍 Ellenőrzés Futtatás Előtt

### 1. Adatok megvannak?
```bash
ls data_processed/
# Kell: train_features.npy, train_labels.npy, test_features.npy, test_filenames.npy
```

### 2. Függőségek telepítve?
```bash
pip install tensorflow opencv-python scikit-learn pandas matplotlib tqdm
```

### 3. GPU működik?
```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 💡 Tippek

### Ha túl lassú:
```bash
# Kisebb modell
python src/train.py --model simple --epochs 30

# Vagy kisebb batch
python src/train.py --model optimized --batch_size 32
```

### Ha túlillesztés van:
```bash
# Regularizált modell
python src/train.py --model regularized --epochs 50

# Vagy kapcsold ki az augmentációt
python src/train.py --model optimized --no_augmentation
```

### Ha nincs GPU:
```bash
# Használj kisebb modellt
python src/train.py --model advanced --epochs 40 --batch_size 32
```

---

## 🎓 Összehasonlítás

Ha szeretnél több modellt összehasonlítani:

```bash
# 1. Simple baseline
python src/train.py --model simple --epochs 30 --run_name "baseline_simple"

# 2. Advanced
python src/train.py --model advanced --epochs 40 --run_name "baseline_advanced"

# 3. Optimized (legjobb)
python src/train.py --model optimized --epochs 60 --run_name "best_optimized"

# 4. Pro-Hybrid (nagy)
python src/train.py --model pro_hybrid --epochs 50 --run_name "big_pro_hybrid"
```

Majd nézd meg a `validation_report.txt` fájlokat!

---

## ⚠️ Gyakori Hibák

### "CUDA out of memory"
```bash
# Csökkentsd a batch size-t
python src/train.py --model optimized --batch_size 32
```

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "FileNotFoundError: data_processed"
```bash
# Futtasd először a preprocessing-et
python src/data_preprocessing.py
```

---

## 📞 Segítség

Ha valami nem működik:
1. Ellenőrizd, hogy az adatok feldolgozva vannak (`data_processed/`)
2. Nézd meg a GPU státuszt
3. Próbáld kisebb modellel/batch-csel
4. Ellenőrizd a függőségeket

