"""
train.py - FAST Training Script for Facial Emotion Recognition
Author: Yash Solanki (241306114)
Department: Computer Applications, SGT University

Optimized for speed: ~5 minutes on CPU
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

from model import build_model
from utils import (
    EMOTION_LABELS, IMG_HEIGHT, IMG_WIDTH,
    plot_training_history, plot_confusion_matrix
)


def download_dataset():
    """Download FER2013 or use local data/ folder."""
    print("=" * 60)
    print("  Loading FER2013 Dataset")
    print("=" * 60)

    local_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if os.path.exists(local_data_path):
        train_dir = os.path.join(local_data_path, 'train')
        test_dir = os.path.join(local_data_path, 'test')
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            print(f"[OK] Found local dataset at: {local_data_path}")
            return local_data_path

    try:
        import kagglehub
        print("Downloading from Kaggle...")
        path = kagglehub.dataset_download("msambare/fer2013")
        print(f"[OK] Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] {e}")
        print(f"\nManual download: https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"Extract train/ and test/ folders into: {local_data_path}")
        raise


def load_data(dataset_path):
    """Load images from train/ and test/ directories."""
    import cv2

    emotion_map = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'sad': 4, 'surprise': 5, 'neutral': 6
    }

    def load_split(split_dir):
        images, labels = [], []
        if not os.path.exists(split_dir):
            return np.array([]), np.array([])

        for emotion_name, label_idx in emotion_map.items():
            emotion_dir = os.path.join(split_dir, emotion_name)
            if not os.path.exists(emotion_dir):
                continue

            count = 0
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                        images.append(img)
                        labels.append(label_idx)
                        count += 1
                except Exception:
                    continue

            print(f"   {emotion_name:>10}: {count} images")

        return np.array(images), np.array(labels)

    print("\nLoading training data...")
    X_train, y_train = load_split(os.path.join(dataset_path, 'train'))

    print("\nLoading test data...")
    X_test, y_test = load_split(os.path.join(dataset_path, 'test'))

    print(f"\nTotal: {len(X_train)} train, {len(X_test)} test")
    return X_train, y_train, X_test, y_test


def train_model():
    """Fast training pipeline - ~5 minutes on CPU."""
    print("=" * 60)
    print("  Facial Emotion Recognition - FAST Training")
    print("  Yash Solanki | 241306114 | BCA, SGT University")
    print("=" * 60)

    # Step 1: Get dataset
    dataset_path = download_dataset()

    # Step 2: Load data
    X_train, y_train, X_test, y_test = load_data(dataset_path)
    if len(X_train) == 0:
        print("[ERROR] No training data found!")
        return

    # Step 3: Preprocess
    print("\nPreprocessing...")
    X_train = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 4: Build model
    print("\nBuilding CNN model...")
    model = build_model()
    model.summary()

    # Step 5: Train - FAST settings (no augmentation, 10 epochs, big batch)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint('emotion_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("\n" + "=" * 60)
    print("  Starting FAST Training (10 epochs, no augmentation)...")
    print("=" * 60)

    BATCH_SIZE = 128   # Larger batch = faster
    EPOCHS = 10        # Fewer epochs = faster

    history = model.fit(
        X_train, y_train,          # Direct training - NO augmentation (much faster)
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Step 6: Evaluate
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   Test Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"   Test Loss:      {test_loss:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTION_LABELS))

    # Step 7: Save plots
    print("\nSaving plots...")
    plot_training_history(history)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(cm)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Model saved to: emotion_model.keras")
    print(f"  Test Accuracy:   {test_accuracy * 100:.2f}%")
    print(f"\n  Next steps:")
    print(f"    python detect_emotion.py    (webcam)")
    print(f"    streamlit run app.py        (web app)")
    print("=" * 60)


if __name__ == '__main__':
    train_model()
