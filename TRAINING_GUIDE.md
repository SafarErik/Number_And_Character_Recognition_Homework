# Tanítás Útmutató - train.py

## Használat

### Alap parancsok

```bash
# Alapértelmezett (Advanced CNN)
python src/train.py

# Optimalizált modell (ajánlott!)
python src/train.py --model optimized

# Egyszerű modell gyors teszteléshez
python src/train.py --model simple

# Összes elérhető modell
python src/train.py --model simple
python src/train.py --model advanced
python src/train.py --model mlp
python src/train.py --model hybrid
python src/train.py --model pro_hybrid
python src/train.py --model regularized
python src/train.py --model optimized
```

---

## Paraméterek

### --model
Választható modellek:
- `simple`: Gyors, alapvető CNN (321K paraméter)
- `advanced`: Fejlettebb CNN BatchNorm-mal (635K paraméter)
- `mlp`: Teljes összeköttetésű háló (303K paraméter)
- `hybrid`: Kettős Conv2D rétegek (2.2M paraméter)
- `pro_hybrid`: Swish aktiváció + Label Smoothing (4.6M paraméter)
- `regularized`: Pro-Hybrid L2 regularizációval (4.6M paraméter)
- **`optimized`** ⭐: Focal Loss + Swish + speciális augmentáció (2.7M paraméter)

### --run_name
Egyedi név a futtatáshoz (alkönyvtár neve a `results/` mappában)

```bash
python src/train.py --model optimized --run_name "kiserlet_01"
# Eredmények: results/kiserlet_01/
```

### --epochs
Maximális epoch szám (default: 50)

```bash
python src/train.py --model optimized --epochs 80
```

### --batch_size
Batch méret (default: 64)

```bash
python src/train.py --model optimized --batch_size 128
```

### --no_augmentation
Kikapcsolja az adatbővítést

```bash
python src/train.py --model simple --no_augmentation
```

---

## Optimalizált Modell Speciális Funkciói ⭐

Az `optimized` modell különleges beállításokkal rendelkezik:

### 1. **Focal Loss**
- Koncentrál a nehéz karakterekre (o/0, w/W, I/l/1)
- Nem pazarol időt az egyszerű esetekre

### 2. **Speciális Augmentáció**
- **Zoom TILTVA** - megőrzi a karakterek méretét (w vs W)
- Kisebb forgatás (8° vs 15°)
- Kisebb eltolás (0.08 vs 0.15)
- Minimális shear (0.05 vs 0.2)

### 3. **Megnövelt Türelem**
- EarlyStopping patience: 12 (vs 10)
- ReduceLROnPlateau patience: 4 (vs 3)
- Focal Loss természetesen hullámzik, ezért kell több türelem

### 4. **Swish Aktiváció**
- Finomabb gradiens mint a ReLU
- Jobb teljesítmény mély hálóknál

---

## Példa Futtatások

### Gyors teszt (5 epoch)
```bash
python src/train.py --model simple --epochs 5 --run_name "gyors_teszt"
```

### Optimalizált modell teljes tanítás
```bash
python src/train.py --model optimized --epochs 60 --run_name "opt_v1"
```

### Pro-Hybrid nagy batch-csel
```bash
python src/train.py --model pro_hybrid --epochs 50 --batch_size 128 --run_name "pro_big_batch"
```

### Size Expert mód (zoom nélkül)
```bash
python src/train.py --model hybrid --run_name "size_expert_hybrid"
# A "size_expert" név automatikusan kikapcsolja a zoom-ot!
```

---

## Eredmények

Minden futtatás létrehoz egy mappát a `results/` könyvtárban:

```
results/
  └── optimized_20231119_140530/  (vagy saját run_name)
      ├── best_model.keras         # Legjobb modell súlyok
      ├── submission.csv            # Beadási fájl
      ├── history.png              # Tanítási görbék
      ├── misclassified.png        # Rossz osztályozások
      └── validation_report.txt    # Teljes riport
```

---

## GPU Támogatás

A kód automatikusan észleli a GPU-t és beállítja a memory growth-ot:

```
GPU(k) észlelve: 1 db
```

Ha nincs GPU, CPU-n fut (lassabb).

---

## Tippek

### Melyik modellt használjam?

| Cél | Modell | Indoklás |
|-----|--------|----------|
| Gyors kísérletezés | `simple` | Leggyorsabb, baseline |
| Legjobb pontosság | `optimized` ⭐ | Focal Loss + Swish |
| Nagy adathalmaz | `pro_hybrid` | Sok paraméter, Label Smoothing |
| Túlillesztés | `regularized` | Erős L2 + Dropout |
| Benchmark | `mlp` | Látjuk mennyit segít a konvolúció |

### Hány epoch-ra van szükségem?

- **Simple/Advanced**: 30-50 epoch
- **Hybrid/Pro**: 40-60 epoch
- **Optimized**: 50-80 epoch (Focal Loss lassabban konvergál, de magasabb pontosságot ér el)

### Mi a különbség az augmentációban?

| Paraméter | Alapértelmezett | Optimized | Magyarázat |
|-----------|----------------|-----------|------------|
| rotation_range | 15° | 8° | Kevesebb forgatás |
| shift | 0.15 | 0.08 | Kevesebb eltolás |
| shear | 0.2 | 0.05 | Minimális ferdeség |
| **zoom** | 0.05 | **0.0** ⚠️ | **TILTVA w/W miatt!** |

---

## Hibaelhárítás

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "OOM (Out of Memory)" hiba
Csökkentsd a batch size-t:
```bash
python src/train.py --model optimized --batch_size 32
```

### Lassú tanítás
- Használj kisebb modellt (`simple`)
- Csökkentsd az epoch számot
- Kapcsold ki az augmentációt (`--no_augmentation`)

