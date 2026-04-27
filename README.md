# Real-Time Facial Emotion Recognition System using CNN

**Deep Learning Project** | Yash Solanki (241306114) | BCA, SGT University

A Facial Emotion Recognition system built using Convolutional Neural Networks (CNN) that detects and classifies 7 human emotions from facial images.

---

## Detectable Emotions

| Emotion | Emotion | Emotion |
|---------|---------|---------|
| Angry | Happy | Surprise |
| Disgust | Sad | Neutral |
| Fear | | |

---

## Project Structure

### Files to Upload on GitHub

```
yash/
├── app.py                  # Streamlit web app (main entry point)
├── model.py                # CNN model architecture
├── utils.py                # Helper functions
├── train.py                # Training script (run locally)
├── detect_emotion.py       # Webcam detection (run locally)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── emotion_model.keras     # Trained model (generated after training)
├── training_history.png    # Training plot (generated after training)
├── confusion_matrix.png    # Confusion matrix (generated after training)
└── sample_images/          # Sample faces for demo
    ├── happy_face.png
    ├── sad_face.png
    └── surprise_face.png
```
---

## Setup & Installation (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install kagglehub

# 2. Train the model (~30 min on CPU)
python train.py

# 3. Run the Streamlit app
streamlit run app.py

# 4. (Optional) Run webcam detection
python detect_emotion.py
```

---

The app will automatically install packages from `requirements.txt` and start.

---

## Model Architecture

```
Input (48x48x1 Grayscale)
  -> Conv2D(32) + BatchNorm + MaxPool + Dropout
  -> Conv2D(64) + BatchNorm + MaxPool + Dropout
  -> Conv2D(128) + BatchNorm + MaxPool + Dropout
  -> Dense(256) + BatchNorm + Dropout
  -> Dense(7, softmax) -> Output (7 emotions)
```

- **Dataset:** FER2013 (35,887 images, 48x48 grayscale)
- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy

---

## References

1. I. J. Goodfellow et al., "Deep Learning," MIT Press, 2016.
2. FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013

---

**Developed by Yash Solanki | Reg. No. 241306114 | BCA, SGT University**
