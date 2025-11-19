# utils.py Változások Összefoglalója

## 🔄 Főbb Változások

### 1. **Egyszerűbb Visszatérési Struktúra**

#### ❌ RÉGI (túl bonyolult):
```python
return (X_train, y_train_cat), (X_val, y_val_cat, y_val_labels), X_test, num_classes, test_filenames
```

#### ✅ ÚJ (egyszerű):
```python
return X_train, X_val, y_train_labels, y_val_labels, X_test, test_filenames, num_classes
```

**Előny**: Nem kell tuple boncolgatás, átláthatóbb kód.

---

### 2. **One-hot Encoding Áthelyezése**

#### ❌ RÉGI (utils.py-ban):
```python
y_train_cat = tf.keras.utils.to_categorical(y_train_labels, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val_labels, num_classes)
```

#### ✅ ÚJ (train.py-ban):
```python
y_train = tf.keras.utils.to_categorical(y_train_labels, num_classes)
y_val = tf.keras.utils.to_categorical(y_val_labels, num_classes)
```

**Előny**: 
- utils.py nem függ TensorFlow-tól
- Rugalmasabb (könnyebb más formátumra váltani)
- Tisztább felelősségek

---

### 3. **Validációs Halmaz Mérete**

#### ❌ RÉGI:
```python
test_size=0.2  # 20% validáció
```

#### ✅ ÚJ:
```python
test_size=0.1  # 10% validáció
```

**Előny**: Több adat a tanításhoz (90% vs 80%).

---

### 4. **Csatorna Dimenzió Ellenőrzés**

#### ✅ ÚJ funkció:
```python
# Csatorna dimenzió ellenőrzése (legyen (N, 32, 32, 1))
if len(X_train_full.shape) == 3:
    X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
if len(X_test.shape) == 3:
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
```

**Előny**: Biztonsági ellenőrzés, ha a preprocessing nem adja vissza a helyes formátumot.

---

### 5. **Float32 Casting**

#### ❌ RÉGI:
```python
X_train_full = X_train_full / 255.0  # implicit float64
```

#### ✅ ÚJ:
```python
X_train_full = X_train_full.astype('float32') / 255.0  # explicit float32
```

**Előny**: 
- Kisebb memóriahasználat (float32 vs float64)
- Gyorsabb GPU számítás
- Kompatibilis a mixed precision training-gel

---

### 6. **Tisztább Hibaüzenetek**

#### ✅ ÚJ:
```python
print(f"HIBA: Nem találhatók a feldolgozott .npy fájlok! Futtasd le először a data_preprocessing.py-t!")
print(f"Részletek: {e}")
```

**Előny**: Azonnal látszik mi a probléma és mit kell tenni.

---

## 📊 Összehasonlítás

| Funkció | RÉGI | ÚJ | Előny |
|---------|------|-----|-------|
| Visszatérés | Bonyolult tuple | Egyszerű lista | Átláthatóbb |
| One-hot | utils.py-ban | train.py-ban | Kevesebb függőség |
| Validáció | 20% | 10% | Több tanító adat |
| Dimenzió check | ❌ | ✅ | Biztonságosabb |
| Float típus | implicit | explicit float32 | Gyorsabb, kevesebb RAM |
| TF függőség | ✅ | ❌ | Tisztább |

---

## 🎯 Miért Jobb Ez?

### 1. **Kevesebb Függőség**
Az `utils.py` már nem függ TensorFlow-tól:
```python
# ❌ RÉGI
import tensorflow as tf

# ✅ ÚJ
# Nincs TensorFlow import!
```

### 2. **Egyszerűbb train.py**
```python
# ✅ ÚJ - egyszerű kicsomagolás
X_train, X_val, y_train_labels, y_val_labels, X_test, test_filenames, num_classes = data

# One-hot encoding itt
y_train = tf.keras.utils.to_categorical(y_train_labels, num_classes)
y_val = tf.keras.utils.to_categorical(y_val_labels, num_classes)
```

### 3. **Memória Optimalizálás**
- **float32** vs float64 → **50% kevesebb RAM**
- 10% validáció vs 20% → **11% több tanító adat**

### 4. **Biztonságosabb**
```python
# Automatikus javítás, ha a dimenzió nem jó
if len(X_train_full.shape) == 3:
    X_train_full = X_train_full.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
```

---

## ✅ Kompatibilitás

Az új utils.py **teljesen kompatibilis** az új train.py-jal:

```python
# train.py automatikusan kezeli
data = load_data()
X_train, X_val, y_train_labels, y_val_labels, X_test, test_filenames, num_classes = data
y_train = tf.keras.utils.to_categorical(y_train_labels, num_classes)
y_val = tf.keras.utils.to_categorical(y_val_labels, num_classes)
```

---

## 📁 Fájlok Státusza

### ✅ Frissítve:
- `src/utils.py` - Egyszerűsítve, TensorFlow mentes
- `src/train.py` - One-hot encoding hozzáadva

### 📝 Nincs szükség módosításra:
- `src/data_preprocessing.py` - Már kompatibilis (4D tömb)
- `src/models.py` - Működik
- `src/visualize.py` - Működik

---

## 🚀 Tesztelés

```bash
cd "C:\_Tomi\_BME\_5.felev\Adatelemzés\HF\Program"

# 1. Preprocessing (ha még nem futott)
python src/data_preprocessing.py

# 2. Tanítás az új kóddal
python src/train.py --model optimized --epochs 5 --run_name "teszt_uj_kod"
```

---

## 💡 Megjegyzés

Az új kód **nem töri el** a meglévő funkcionalitást, csak:
- Egyszerűbbé teszi
- Gyorsabbá teszi (float32)
- Biztonságosabbá teszi (dimenzió check)
- Több adatot ad a tanításhoz (10% vs 20% validáció)

