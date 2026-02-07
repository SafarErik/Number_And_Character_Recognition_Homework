import tensorflow as tf
from keras.src.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50


def build_simple_cnn(input_shape, num_classes):
    """Builds a simple Keras CNN model."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_advanced_cnn(input_shape, num_classes):
    """Builds a more advanced Keras CNN with Batch Normalization and more layers."""
    model = Sequential()

    # First Block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Second Block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Third Block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Dense Block
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_keras_mlp(input_shape, num_classes):
    """Builds a Keras-based MLP for comparison with CNNs."""
    # input_shape is (28, 28, 1), needs to be flattened first
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_hybrid_cnn(input_shape, num_classes):
    """Builds a hybrid CNN with larger kernels initially and deeper structure."""
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

    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_pro_hybrid_cnn(input_shape, num_classes):
    """
    Enhanced Hybrid model with Swish activation and Label Smoothing.
    """
    model = Sequential()

    # --- BLOCK 1 ---
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- BLOCK 2 ---
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- BLOCK 3 ---
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

    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model


def build_regularized_hybrid_cnn(input_shape, num_classes):
    """
    Optimized 3-block model with L2 regularization.
    """
    model = Sequential()
    reg = regularizers.l2(0.0005)

    # --- BLOCK 1 ---
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape,
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- BLOCK 2 ---
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))

    # --- BLOCK 3 ---
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    # --- OUTPUT ---
    model.add(Flatten())

    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Dropout(0.55))

    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def build_resnet_transfer(input_shape, num_classes):
    """
    ResNet50 Transfer Learning.
    Input: (32, 32, 3) - Needs RGB!
    """
    # Load ImageNet weights, exclude top
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Option 2 (Better): Allow fine-tuning of last few blocks
    base_model.trainable = True
    # Use very small LR to avoid destroying weights!

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(num_classes),
        Activation('softmax', dtype='float32')
    ])

    # Small learning rate for fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_deep_hybrid_cnn(input_shape, num_classes):
    """
    Deep network optimized for 64x64 images. 4 Blocks.
    "Deep Expert".
    """
    model = Sequential()

    # Mild L2 regularization against overfitting
    reg = regularizers.l2(0.0005)

    # --- BLOCK 1 (64 -> 32) ---
    # Larger kernel (5x5) to capture shapes
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape,
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Size halves to 32

    # --- BLOCK 2 (32 -> 16) ---
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Size halves to 16

    # --- BLOCK 3 (16 -> 8) ---
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Size halves to 8

    # --- BLOCK 4 (NEW! 8 -> 4) ---
    # Needed for 64 size
    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(MaxPooling2D((2, 2)))  # Size halves to 4

    # --- OUTPUT ---
    model.add(GlobalMaxPooling2D())

    model.add(Dropout(0.6))  # Strong dropout for safety

    model.add(Dense(num_classes))
    model.add(Activation('softmax', dtype='float32'))

    # Label smoothing for clearer boundaries
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Cautious LR for deeper net + Clipnorm for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model