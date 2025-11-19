# Elérhető Modellek - Összefoglaló

## Modellarchitektúrák

A projektben **7 különböző CNN/MLP modell** található az OCR feladatra:

### 1. **build_simple_cnn** (321,854 paraméter)
- Egyszerű alap CNN modell
- 2 Conv2D réteg + Dense
- Jó kiindulópont és baseline

### 2. **build_advanced_cnn** (635,070 paraméter)
- Fejlettebb CNN BatchNormalization-nel
- 3 Conv2D blokk
- Jobb generalizáció

### 3. **build_keras_mlp** (303,294 paraméter)
- Teljes összeköttetésű (MLP) háló
- Összehasonlításhoz használható
- Nem használ konvolúciót

### 4. **build_hybrid_cnn** (2,255,006 paraméter)
- Kettős Conv2D rétegek minden blokkban
- Hangsúly a kis/nagybetűk megkülönböztetésén
- BatchNorm + Dropout

### 5. **build_pro_hybrid_cnn** (4,648,350 paraméter)
- Hybrid modell turbózva
- **Swish aktiváció** (finomabb gradiens)
- **Label Smoothing** (0.1)
- He normal inicializáció

### 6. **build_optimized_ocr_model** ⭐ (2,714,718 paraméter) **ÚJ!**
- **Focal Loss** használata (nehéz karakterek: 'o' vs '0', 'w' vs 'W')
- **Swish aktiváció**
- BatchNormalization minden rétegnél (gyorsabb tanulás)
- 4 Conv2D blokk fokozatosan növekvő filterekkel (32→64→128→256)
- Flatten (térbeli információ megőrzése)
- 512-es Dense réteg
- **Optimális egyensúly**: sebesség + pontosság

### 7. **build_regularized_hybrid_cnn** (4,648,350 paraméter)
- Pro-Hybrid modell **L2 regularizációval**
- Túlillesztés ellen
- Label Smoothing
- Erősebb Dropout (0.55)

---

## Focal Loss ✨

Az új `categorical_focal_loss()` függvény segít a nehéz esetek kezelésében:
- **Gamma**: Fókuszálás erőssége (default: 2.0)
- **Alpha**: Súlyozás (default: 0.25)
- Automatikusan több súlyt ad a rosszul osztályozott példányokra

---

## Használat

```python
from src.models import build_optimized_ocr_model

# Modell létrehozása
model = build_optimized_ocr_model(
    input_shape=(32, 32, 1),  # Szürkeárnyalatos 32x32-es képek
    num_classes=62             # 0-9, a-z, A-Z
)

# A modell már compile-olva van Focal Loss-szal és Adam optimizerrel
# Kész a tanításra!
```

---

## Melyiket válasszam?

| Feladat | Javasolt Modell |
|---------|----------------|
| Gyors tesztelés | `build_simple_cnn` |
| Legjobb pontosság (lassabb) | `build_pro_hybrid_cnn` |
| **Sebesség + Pontosság** ⭐ | `build_optimized_ocr_model` |
| Túlillesztés problémánál | `build_regularized_hybrid_cnn` |
| Benchmark | `build_keras_mlp` |

---

## Tanítás tippek

- **Focal Loss**: Különösen hasznos hasonló karaktereknél (o/0, I/l/1, w/W)
- **BatchNormalization**: Gyorsabb konvergencia
- **Swish aktiváció**: Általában jobb mint ReLU mély hálóknál
- **Label Smoothing**: Megakadályozza a túlzott magabiztosságot

