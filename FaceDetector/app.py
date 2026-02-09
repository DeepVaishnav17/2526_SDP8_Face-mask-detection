import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mask Detector AI", layout="wide", page_icon="😷")

# --- FIX FOR KERAS 3 INCOMPATIBILITY ---
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' which causes the first TypeError
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# --- CUSTOM UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    h1 { color: #1E3A8A; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_resources():
    # 1. Load the model with custom layer fix AND safe_mode=False
    # safe_mode=False prevents the "2 input tensors" error
    model = load_model(
        "keras_model.h5", 
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
        compile=False,
        safe_mode=False
    )
    
    # 2. Force the model to build with a single input shape
    model.build((None, 224, 224, 3))
    
    # 3. Load class labels
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 4. Load face detector
    face_net = cv2.dnn.readNet("face_detector/deploy.prototxt", 
                               "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    
    return model, class_names, face_net

# Initialize resources
try:
    model, class_names, face_net = load_resources()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Try downgrading: pip install tensorflow==2.15.0")
    st.stop()

# --- PREDICTION LOGIC ---
def predict_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0: continue
            
            face = cv2.resize(face, (224, 224))
            face = np.asarray(face, dtype=np.float32).reshape(1, 224, 224, 3)
            face = (face / 127.5) - 1 
            
            # Predict
            prediction = model.predict(face, verbose=0)
            index = np.argmax(prediction)
            class_name = class_names[index]
            score = prediction[0][index]

            color = (0, 255, 0) if "mask" in class_name.lower() or "0" in class_name else (255, 0, 0)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", 
                        (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

# --- UI LAYOUT ---
st.title("🛡️ Smart AI Face Mask Detector")

option = st.sidebar.selectbox("Choose Input:", ("Image Upload", "Live Webcam"))

if option == "Image Upload":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        frame = np.array(img)
        processed = predict_mask(frame.copy())
        st.image(processed, caption="Detection Result", use_container_width=True)

elif option == "Live Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    vid = cv2.VideoCapture(0)

    while run:
        ret, frame = vid.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = predict_mask(frame)
        FRAME_WINDOW.image(processed)
    else:
        vid.release()