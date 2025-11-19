import tensorflow as tf
from keras.src.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)

from tensorflow.keras import regularizers


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
    A Hybrid modell felturbózva: Swish aktiváció, GAP és Label Smoothing.
    """
    model = Sequential()

    # --- 1. BLOKK (Swish-sel) ---
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

    model.add(Flatten())

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
    A Pro-Hybrid modell L2 Regularizációval a túlillesztés ellen.
    """
    model = Sequential()

    # Regularizációs erősség
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

    # Extra réteg
    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    # --- KIMENET ---
    model.add(Flatten())

    # Dense réteg regularizációval
    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    # Kicsit erősebb Dropout
    model.add(Dropout(0.55))

    model.add(Dense(num_classes, activation='softmax'))

    # Label smoothing
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model