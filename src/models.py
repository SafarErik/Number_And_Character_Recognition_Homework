from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)
from sklearn.neural_network import MLPClassifier


def build_mlp(num_classes):
    """Épít egy sklearn MLP modellt."""
    # Az sklearn modellnek más a 'fit' és 'predict' API-ja,
    # ezért érdemes lehet egy Keras Dense hálóval helyettesíteni
    # a könnyebb összehasonlításért.
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, verbose=True)
    return model  # Figyelem: ez nem Keras modell!


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