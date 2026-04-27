"""
utils.py - Helper functions for Facial Emotion Recognition
Author: Yash Solanki (241306114)
Department: Computer Applications, SGT University
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Emotion labels and image dimensions
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_HEIGHT = 48
IMG_WIDTH = 48


def preprocess_face(face_image):
    """Preprocess a face image for model prediction."""
    if len(face_image.shape) == 3:
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_image

    face_resized = cv2.resize(face_gray, (IMG_WIDTH, IMG_HEIGHT))
    face_normalized = face_resized / 255.0
    face_ready = face_normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    return face_ready


def get_emotion_prediction(model, face_image):
    """Predict emotion from a face image."""
    processed = preprocess_face(face_image)
    predictions = model.predict(processed, verbose=0)[0]
    emotion_index = np.argmax(predictions)
    emotion_label = EMOTION_LABELS[emotion_index]
    confidence = float(predictions[emotion_index]) * 100
    return emotion_label, confidence, predictions


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Training', color='#2196F3', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', color='#FF5722', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Training', color='#2196F3', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', color='#FF5722', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training history saved to: {save_path}")


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
           title='Confusion Matrix', ylabel='True Emotion', xlabel='Predicted Emotion')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > threshold else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved to: {save_path}")


def draw_emotion_on_frame(frame, face_coords, emotion, confidence):
    """Draw bounding box and emotion label on a video frame."""
    x, y, w, h = face_coords

    emotion_colors = {
        'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Angry': (0, 0, 255),
        'Surprise': (0, 255, 255), 'Fear': (255, 165, 0),
        'Disgust': (128, 0, 128), 'Neutral': (200, 200, 200)
    }
    color = emotion_colors.get(emotion, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{emotion} ({confidence:.1f}%)"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame
