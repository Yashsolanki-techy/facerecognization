"""
model.py - CNN Model Architecture for Facial Emotion Recognition
Author: Yash Solanki (241306114)
Department: Computer Applications, SGT University

Architecture: Lightweight CNN optimized for fast training
    Input (48x48x1) -> 3 Conv Blocks -> Dense(256) -> Dense(7, softmax)
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, Input
)


def build_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Build a lightweight CNN model for fast training.

    Architecture:
    - 3 convolutional blocks (32 -> 64 -> 128 filters)
    - 1 dense layer (256 units)
    - Output: 7 emotions with softmax

    Args:
        input_shape: (48, 48, 1) grayscale image
        num_classes: 7 emotion categories

    Returns:
        Compiled Keras model
    """

    model = Sequential(name='FER_CNN')

    # BLOCK 1: Basic edge detection (32 filters)
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))       # 48x48 -> 24x24
    model.add(Dropout(0.25))

    # BLOCK 2: Pattern detection (64 filters)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))       # 24x24 -> 12x12
    model.add(Dropout(0.25))

    # BLOCK 3: High-level features (128 filters)
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))       # 12x12 -> 6x6
    model.add(Dropout(0.25))

    # Classification head
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    print("=" * 60)
    print("  Facial Emotion Recognition - CNN Architecture")
    print("=" * 60)
    model = build_model()
    model.summary()
    print(f"\nTotal Parameters: {model.count_params():,}")
