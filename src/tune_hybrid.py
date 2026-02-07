import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras import regularizers
import keras_tuner as kt
from utils import load_data_for_training_and_prediction as load_data, IMG_SIZE

# Global variable to pass num_classes to the builder function
NUM_CLASSES = 63  # Default fallback


# --- TUNABLE MODEL DEFINITION ---
def build_tunable_model(hp):
    """
    This is the Regularized Hybrid model, but using 'hp' to select values.
    """
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    # --- TUNABLE PARAMETERS ---
    # 1. L2 Regularization (1e-3, 5e-4, 1e-4)
    l2_value = hp.Choice('l2_rate', values=[1e-3, 5e-4, 1e-4])
    reg = regularizers.l2(l2_value)

    # 2. Dropout (0.3 - 0.6)
    dropout_rate = hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1)

    # 3. Learning Rate (Most important!)
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # --- BUILD MODEL (Same as in models.py) ---
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape,
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D((2, 2)))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(MaxPooling2D((2, 2)))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    # Output
    model.add(Flatten())

    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('swish'))

    model.add(Dropout(dropout_rate))  # Using tuner value

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Label smoothing remains fixed 0.1
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Clipnorm remains fixed 1.0 for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def main():
    global NUM_CLASSES
    print("Loading data for tuning...")
    data = load_data()
    if data is None: return

    # We only need validation data for evaluation
    (X_train, y_train), (X_val, y_val, _), _, num_classes, _ = data

    # Update global variable
    NUM_CLASSES = num_classes
    print(f"Detected {NUM_CLASSES} classes.")

    # --- TUNER SETUP ---
    tuner = kt.Hyperband(
        build_tunable_model,
        objective='val_accuracy',
        max_epochs=12,  # 12 is enough to see potential
        factor=3,
        hyperband_iterations=1,
        directory='tuning_dir',
        project_name='hybrid_cnn_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    print("\n--- STARTING SEARCH... This may take a while! ---")

    # Use same batch_size as intended for final training (32)
    tuner.search(X_train, y_train,
                 epochs=12,
                 validation_data=(X_val, y_val),
                 callbacks=[stop_early],
                 batch_size=32)

    # --- RESULTS ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n" + "=" * 30)
    print("   BEST HYPERPARAMETERS")
    print("=" * 30)
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"L2 Rate:       {best_hps.get('l2_rate')}")
    print(f"Dropout:       {best_hps.get('dropout')}")
    print("=" * 30)
    print("\nPlease update src/models.py and src/train_kfold.py")
    print("with these values before running the full training!")


if __name__ == "__main__":
    main()