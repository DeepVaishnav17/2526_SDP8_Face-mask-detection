import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "mask_detector.h5"
IMAGE_PATH = "img.webp"

model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype("float32")

# 🔴 IMPORTANT FIX
img = preprocess_input(img)

img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0]

print("Raw prediction:", pred)

if pred[0] > pred[1]:
    print("😷 WITH MASK")
else:
    print("❌ NO MASK")
