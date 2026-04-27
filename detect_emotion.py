"""
detect_emotion.py - Real-Time Webcam Emotion Detection
Author: Yash Solanki (241306114)
Department: Computer Applications, SGT University

This script uses your trained CNN model to detect emotions
in real-time from your webcam feed.

It works in 3 steps:
1. Capture video from webcam
2. Detect faces using Haar Cascade (OpenCV)
3. Predict emotion for each face using the CNN model

Usage:
    python detect_emotion.py

Controls:
    Press 'q' to quit the webcam feed
"""

import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import (
    EMOTION_LABELS, preprocess_face, draw_emotion_on_frame
)


def run_webcam_detection():
    """
    Start real-time emotion detection from webcam.
    """
    print("=" * 60)
    print("  🎥 Real-Time Facial Emotion Detection")
    print("  👤 Yash Solanki | 241306114 | BCA, SGT University")
    print("=" * 60)
    
    # ----- Step 1: Load the trained model -----
    MODEL_PATH = 'emotion_model.keras'
    print(f"\n📦 Loading model from: {MODEL_PATH}")
    
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Please run 'python train.py' first to train the model.")
        return
    
    # ----- Step 2: Load Face Detector -----
    # Haar Cascade is a pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print("✅ Face detector loaded!")
    
    # ----- Step 3: Start Webcam -----
    print("\n🎥 Starting webcam... (Press 'q' to quit)")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        print("💡 Make sure your webcam is connected and not used by another app.")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        # Read a frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from webcam")
            break
        
        frame_count += 1
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # How much to reduce image size at each scale
            minNeighbors=5,      # Minimum neighbors for a detection to be valid
            minSize=(48, 48)     # Minimum face size to detect
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from grayscale image
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess the face for our model
            processed_face = preprocess_face(face_roi)
            
            # Predict emotion
            predictions = model.predict(processed_face, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion_label = EMOTION_LABELS[emotion_idx]
            confidence = predictions[emotion_idx] * 100
            
            # Draw bounding box and label on the frame
            frame = draw_emotion_on_frame(
                frame, (x, y, w, h), emotion_label, confidence
            )
        
        # Add title and info text to the frame
        cv2.putText(frame, "Facial Emotion Recognition - Yash Solanki",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces Detected: {len(faces)} | Press 'q' to quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        
        # Display the frame
        cv2.imshow('Facial Emotion Recognition', frame)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Webcam closed. Processed {frame_count} frames.")


# Run webcam detection when this script is executed
if __name__ == '__main__':
    run_webcam_detection()
