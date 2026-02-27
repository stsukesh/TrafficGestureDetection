import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- CONFIG & LOADING ---
st.set_page_config(page_title="AI Traffic Gesture Detector", layout="centered")
st.title("🚦 Traffic Signal Gesture Recognition")
st.write("Upload a photo of a traffic officer to predict the signal.")

@st.cache_resource
def load_ai_model():
    return load_model('gesture_model.h5')

model = load_ai_model()
classes = ['No Gesture', 'Move Left', 'Move Right', 'Start', 'Hand Down', 'Stop Front']

# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

# --- UI: FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Drawing logic for the UI
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Feature Extraction (54 features)
        landmarks = []
        for i in range(18):
            lm = results.pose_landmarks.landmark[i]
            landmarks.extend([lm.x, lm.y, lm.z])
        
        input_data = np.array(landmarks).reshape(1, 54)
        
        # Prediction
        prediction = model.predict(input_data)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        label = classes[class_id] if class_id < len(classes) else f"Class {class_id}"

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_rgb, caption="Original Image", width="stretch")        
        with col2:
            st.image(annotated_image, caption="AI Pose Detection", width="stretch")
        st.success(f"### Prediction: **{label}**")
        st.progress(float(confidence))
        st.write(f"Confidence: {int(confidence*100)}%")
        
    else:
        st.error("AI could not detect a human pose in this image. Please try another.")                                         