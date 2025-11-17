import numpy as np
import tensorflow as tf
from utils import load_data, save_evaluation
from models import build_simple_cnn
from tensorflow.keras.callbacks import EarlyStopping

# --- Konfiguráció ---
MODEL_TO_RUN = 'simmpe' # Ezt változtasd: 'simple' vagy 'advanced'
MODEL_NAME = 'cnn_advanced_v1'
EPOCHS = 100
BATCH_SIZE = 64

# --- 1. Adatok betöltése ---
print("Adatok betöltése...")
(X_train, y_train), (X_val, y_val), (X_test, y_test_one_hot), num_classes = load_data(model_type='cnn')
input_shape = X_train.shape[1:] # (28, 28, 1)

# --- 2. Modell kiválasztása és építése ---
print(f"Modell építése: {MODEL_TO_RUN}")
if MODEL_TO_RUN == 'simple':
    model = build_simple_cnn(input_shape, num_classes)
else:
    raise ValueError("Ismeretlen modell típus")

model.summary()

# --- 3. Tanítás ---
early_stopper = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

print("Tanítás indítása...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopper]
)

# --- 4. Kiértékelés és mentés ---
print("Kiértékelés a teszt adatokon...")
# A kiértékeléshez szükségünk van az eredeti (nem one-hot) címkékre
y_test_labels = np.argmax(y_test_one_hot, axis=1)
save_evaluation(history, model, X_test, y_test_labels, model_name=MODEL_NAME)

# Modell elmentése későbbi használatra
model.save(f'results/{MODEL_NAME}.h5')
print(f"Modell elmentve: results/{MODEL_NAME}.h5")