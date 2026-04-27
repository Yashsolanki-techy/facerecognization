"""
app.py - Streamlit Web App for Facial Emotion Recognition
Author: Yash Solanki (241306114)
Department: Computer Applications, SGT University
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import io

st.set_page_config(
    page_title="Facial Emotion Recognition | Yash Solanki",
    page_icon="FER",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
    .main-header p { font-size: 1rem; opacity: 0.9; }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        margin: 1rem 0; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .emotion-card h2 { font-size: 2rem; margin: 0; }
    .emotion-card p { font-size: 1.1rem; opacity: 0.9; }
    .info-card {
        background: #f8f9fa; padding: 1.5rem; border-radius: 10px;
        border-left: 4px solid #1a237e; margin: 0.5rem 0;
        color: #222 !important;
    }
    .info-card h4, .info-card p, .info-card li, .info-card b {
        color: #222 !important;
    }
    .sidebar-info {
        background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        color: #222 !important;
    }
    .sidebar-info b, .sidebar-info br {
        color: #222 !important;
    }
    .footer {
        text-align: center; padding: 1.5rem; color: #ccc; font-size: 0.95rem;
        margin-top: 2rem; border-top: 1px solid #333;
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        border-radius: 10px; color: white;
    }
</style>
""", unsafe_allow_html=True)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_ICONS = {
    'Angry': 'Angry', 'Disgust': 'Disgust', 'Fear': 'Fear',
    'Happy': 'Happy', 'Sad': 'Sad', 'Surprise': 'Surprise', 'Neutral': 'Neutral'
}


@st.cache_resource
def load_emotion_model():
    """Load the trained CNN model (cached)."""
    from tensorflow.keras.models import load_model
    model_path = 'emotion_model.keras'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None


def predict_emotion(model, image):
    """Detect faces and predict emotions from an image."""
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    results = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        predictions = model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[emotion_idx]) * 100

        color = (0, 255, 0)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(img_bgr, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'predictions': predictions,
            'bbox': (x, y, w, h)
        })

    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_result, results


def get_sample_images():
    """Load sample images from sample_images/ folder."""
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_images')
    samples = {}
    if os.path.exists(sample_dir):
        for f in sorted(os.listdir(sample_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                name = os.path.splitext(f)[0].replace('_', ' ').title()
                samples[name] = os.path.join(sample_dir, f)
    return samples


def main():
    """Main Streamlit app."""

    # --- HEADER ---
    st.markdown("""
    <div class="main-header">
        <h1>Facial Emotion Recognition System</h1>
        <p>Deep Learning CNN Model | Real-Time Emotion Detection</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem;">
            Developed by Yash Solanki | Reg. No. 241306114 | BCA, SGT University
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("## Project Info")
        st.markdown("""
        <div class="sidebar-info">
            <b>Title:</b> Real-Time Facial Emotion Recognition using CNN<br>
            <b>Student:</b> Yash Solanki<br>
            <b>Reg No:</b> 241306114<br>
            <b>Dept:</b> Computer Applications<br>
            <b>Faculty:</b> SOET, SGT University
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## Detectable Emotions")
        for emotion in EMOTION_LABELS:
            st.markdown(f"- **{emotion}**")

        st.markdown("---")
        st.markdown("## Model Architecture")
        st.markdown("""
        - **Input:** 48x48 Grayscale
        - **Conv Blocks:** 3 (32, 64, 128)
        - **Dense Layers:** 256 -> 7
        - **Optimizer:** Adam
        - **Loss:** Categorical Cross-Entropy
        """)

    # --- LOAD MODEL ---
    model = load_emotion_model()

    if model is None:
        st.error("Model not found! Please run `python train.py` first to train the model.")
        st.info("```bash\npip install -r requirements.txt\npython train.py\nstreamlit run app.py\n```")
        return

    st.success("Model loaded successfully!")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["Upload & Detect", "Model Details", "About"])

    # --- Tab 1: Upload & Detect ---
    with tab1:
        st.markdown("### Upload an Image or Try a Sample")
        st.markdown("Upload a photo with visible faces, or select a sample image below.")

        # Sample images section
        sample_images = get_sample_images()
        image = None

        if sample_images:
            st.markdown("#### Try a Sample Image")
            cols = st.columns(len(sample_images))
            for i, (name, path) in enumerate(sample_images.items()):
                with cols[i]:
                    sample_img = Image.open(path)
                    st.image(sample_img, caption=name, use_container_width=True)
                    if st.button(f"Use {name}", key=f"sample_{i}"):
                        image = sample_img

            st.markdown("---")

        st.markdown("#### Or Upload Your Own")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload a clear photo with visible faces for best results"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

        if image is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)

            with st.spinner("Detecting faces and predicting emotions..."):
                result_image, results = predict_emotion(model, image)

            with col2:
                st.markdown("#### Detection Result")
                st.image(result_image, use_container_width=True)

            if len(results) == 0:
                st.warning("No faces detected. Please try a clearer photo with visible faces.")
            else:
                st.markdown(f"### Found **{len(results)} face(s)**")

                for i, result in enumerate(results):
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h2>{result['emotion']}</h2>
                        <p>Confidence: {result['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Face {i+1} - Detailed Predictions:**")
                    for j, emotion in enumerate(EMOTION_LABELS):
                        prob = float(result['predictions'][j]) * 100
                        st.progress(float(result['predictions'][j]),
                                   text=f"{emotion}: {prob:.1f}%")

                    st.markdown("---")

    # --- Tab 2: Model Details ---
    with tab2:
        st.markdown("### CNN Model Architecture")

        st.markdown("""
        <div class="info-card">
            <h4>How the Model Works</h4>
            <p>The CNN processes facial images through multiple layers to extract features and classify emotions:</p>
            <ol>
                <li><b>Input:</b> 48x48 pixel grayscale face image</li>
                <li><b>Feature Extraction:</b> 3 convolutional blocks detect patterns (edges, shapes, expressions)</li>
                <li><b>Classification:</b> Fully connected layer maps features to 7 emotions</li>
                <li><b>Output:</b> Probability for each emotion (softmax)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Model Summary")
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        st.code(stream.getvalue(), language='text')

        st.markdown("#### Training Performance")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('training_history.png'):
                st.image('training_history.png', caption='Training History', use_container_width=True)
            else:
                st.info("Training history plot not available. Run `python train.py` to generate.")
        with col2:
            if os.path.exists('confusion_matrix.png'):
                st.image('confusion_matrix.png', caption='Confusion Matrix', use_container_width=True)
            else:
                st.info("Confusion matrix not available. Run `python train.py` to generate.")

    # --- Tab 3: About ---
    with tab3:
        st.markdown("### About This Project")

        st.markdown("""
        <div class="info-card">
            <h4>Project Details</h4>
            <p><b>Title:</b> Development of a Real-Time Facial Emotion Recognition System
            using Convolutional Neural Networks (CNN)</p>
            <p><b>Student:</b> Yash Solanki (Reg. No. 241306114)</p>
            <p><b>Department:</b> Computer Applications, SOET</p>
            <p><b>University:</b> SGT University</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### Problem Statement
        This project develops an automated Facial Emotion Recognition (FER) system
        that identifies key emotions (Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust)
        from facial images using Convolutional Neural Networks.

        #### Methodology
        1. **Data Collection & Preprocessing:** FER2013 dataset with normalization and grayscale conversion
        2. **CNN Model Architecture:** Sequential CNN with 3 convolutional blocks for feature extraction
        3. **Model Training & Optimization:** Adam optimizer and Categorical Cross-Entropy loss
        4. **Real-time Inference:** OpenCV integration for face detection and emotion prediction

        #### References
        1. A. K. Jain, M. N. Murty, and P. J. Flynn, "Data clustering: a review," ACM Computing Surveys, 1999.
        2. I. J. Goodfellow, et al., "Deep Learning," MIT Press, 2016.
        """)

    # --- FOOTER ---
    st.markdown("""
    <div class="footer">
        <b>Facial Emotion Recognition System</b><br>
        Developed by <b>Yash Solanki</b> | Reg. No. <b>241306114</b><br>
        BCA, Department of Computer Applications, SOET, SGT University<br>
        Deep Learning Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
