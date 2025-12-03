import tensorflow as tf
from keras.src.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)

from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50


def build_simple_cnn(input_shape, num_classes):
    """Épít egy egyszerű Keras CNN modellt."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_advanced_cnn(input_shape, num_classes):
    """Épít egy fejlettebb Keras CNN-t Batch Norm-mal és több réteggel."""
    model = Sequential()

    # Első blokk
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Második blokk
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Harmadik blokk
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Sűrű (Dense) blokk
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_keras_mlp(input_shape, num_classes):
    """Épít egy Keras-alapú MLP-t, hogy összehasonlítható legyen a CNN-ekkel."""
    # Az input_shape itt (28, 28, 1), először ki kell lapítani
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_hybrid_cnn(input_shape, num_classes):
    """Épít egy fejlettebb Keras CNN-t Batch Norm-mal és több réteggel, és figyel a kis/nagybetűkre"""
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_pro_hybrid_cnn(input_shape, num_classes):
    """
    A Hybrid modell felturbózva: Swish aktiváció és Label Smoothing.
    """
    model = Sequential()

    # --- 1. BLOKK ---
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- 2. BLOKK ---
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- 3. BLOKK ---
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(256, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))


    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model


def build_regularized_hybrid_cnn(input_shape, num_classes):
    """
    A 32x32-re optimalizált 3 blokkos modell (Visszaállítva).
    """
    model = Sequential()
    reg = regularizers.l2(0.0005)

    # --- 1. BLOKK ---
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape,
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- 2. BLOKK ---
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- 3. BLOKK ---
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    # --- KIMENET ---
    model.add(Flatten())

    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Dropout(0.55))

    model.add(Dense(num_classes, activation='softmax'))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def build_resnet_transfer(input_shape, num_classes):
    """
    ResNet50 transzfer tanulás.
    Bemenet: (32, 32, 3) - RGB kell neki!
    """
    # Betöltjük az ImageNet súlyokat, de a "fej" (top) nélkül
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 1. opció: Befagyasztjuk az egészet (csak a mi rétegeink tanulnak)
    # base_model.trainable = False

    # 2. opció (Jobb): Engedjük finomhangolni az utolsó pár blokkot
    base_model.trainable = True
    # De hogy ne rontsa el a súlyokat rögtön, nagyon kicsi LR kell majd!

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # A ResNethez ez illik a legjobban
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Kicsi learning rate a finomhangoláshoz
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_deep_hybrid_cnn(input_shape, num_classes):
    """
    Kifejezetten 64x64-es képekhez: 4 blokkból álló, mélyebb hálózat.
    Ez a "Deep Expert".
    """
    model = Sequential()

    # Enyhe L2 regularizáció a túltanulás ellen
    reg = regularizers.l2(0.0005)

    # --- 1. BLOKK (64 -> 32) ---
    # Nagyobb kernel (5x5) a formák megragadásához
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape,
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Itt feleződik a méret 32-re

    # --- 2. BLOKK (32 -> 16) ---
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Itt feleződik 16-ra

    # --- 3. BLOKK (16 -> 8) ---
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Itt feleződik 8-ra

    # --- 4. BLOKK (ÚJ! 8 -> 4) ---
    # Ez a blokk kell a 64-es méret miatt!
    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Itt feleződik 4-re

    # --- KIMENET ---
    model.add(Flatten())  # Most már 4x4x256 = 4096 bemenet érkezik, ami kezelhető

    # Kicsit nagyobb Dense réteg (512), mert több az infó
    model.add(Dense(512, kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Dropout(0.6))  # Erős dropout a biztonságért

    model.add(Dense(num_classes, activation='softmax'))

    # Label smoothing a biztosabb határokért
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Kicsit óvatosabb LR a mélyebb hálóhoz + Clipnorm a stabilitásért
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model