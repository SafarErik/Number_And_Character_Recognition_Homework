import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras import regularizers
import keras_tuner as kt
import datetime

# Importok a projektedből
from utils import load_data_for_training_and_prediction as load_data, IMG_SIZE


# --- A HANGOLHATÓ MODELL DEFINÍCIÓJA ---
def build_tunable_model(hp):
    """
    Ez ugyanaz a Regularized Hybrid modell, csak a fix számok helyett
    a 'hp' objektumtól kérünk értékeket.
    """
    input_shape = (IMG_SIZE, IMG_SIZE, 1)  # 32 vagy 64, amit épp használsz
    num_classes = 63  # Vagy amennyi a datasetedben van (automatikusan detektáljuk majd)

    # --- HANGOLANDÓ PARAMÉTEREK ---
    # 1. L2 Regularizáció (1e-3, 5e-4, 1e-4)
    l2_value = hp.Choice('l2_rate', values=[1e-3, 5e-4, 1e-4])
    reg = regularizers.l2(l2_value)

    # 2. Dropout (0.3 - 0.6)
    dropout_rate = hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1)

    # 3. Learning Rate (A legfontosabb!)
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # --- MODELL FELÉPÍTÉSE (Ugyanaz, mint a models.py-ban) ---
    model = Sequential()

    # 1. Blokk
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape,
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D((2, 2)))

    # 2. Blokk
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D((2, 2)))

    # 3. Blokk
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    # Kimenet
    model.add(Flatten())

    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Dropout(dropout_rate))  # <-- Itt használjuk a változót

    model.add(Dense(num_classes, activation='softmax'))

    # Label smoothing marad fix 0.1
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Clipnorm is marad fix 1.0 a stabilitásért
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def main():
    print("Adatok betöltése a hangoláshoz...")
    data = load_data()
    if data is None: return

    # Csak a validációs adatok kellenek a kiértékeléshez
    (X_train, y_train), (X_val, y_val, _), _, num_classes, _ = data

    # Hack: A build_model függvénynek át kell adni a num_classes-t,
    # de a Keras Tuner csak a 'hp'-t adja át.
    # Ezért egy "wrapper" (csomagoló) függvényt használunk, vagy
    # egyszerűen globálisan beállítjuk a num_classes-t a build_model-ben (itt most kézzel írtam be fent 63-ra).
    # A legszebb megoldás egy osztály lenne, de a fenti kód működik, ha a num_classes=63 (vagy amennyi nálad).

    # --- TUNER BEÁLLÍTÁSA ---
    tuner = kt.Hyperband(
        build_tunable_model,
        objective='val_accuracy',
        max_epochs=12,  # Nem kell 50, elég 12 epoch, hogy lássuk, melyik indul jól
        factor=3,
        hyperband_iterations=1,
        directory='tuning_dir',
        project_name='hybrid_cnn_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    print("\n--- KERESÉS INDÍTÁSA... Ez eltarthat egy ideig! ---")

    # A batch_size legyen ugyanaz, mint amit majd használni akarsz (32)
    tuner.search(X_train, y_train,
                 epochs=12,
                 validation_data=(X_val, y_val),
                 callbacks=[stop_early],
                 batch_size=32)

    # --- EREDMÉNYEK ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n" + "=" * 30)
    print("   A LEGJOBB BEÁLLÍTÁSOK")
    print("=" * 30)
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"L2 Rate:       {best_hps.get('l2_rate')}")
    print(f"Dropout:       {best_hps.get('dropout')}")
    print("=" * 30)
    print("\nMost írd be ezeket az értékeket a src/models.py fájlba,")
    print("és a src/train_kfold.py fájlba, mielőtt elindítod a nagy tanítást!")


if __name__ == "__main__":
    main()