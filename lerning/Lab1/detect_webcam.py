# detect_webcam.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import config_d  # Import your config for consistent paths


print("[INFO] Loading face detector (MediaPipe)...")
mp_face_detection = mp.solutions.face_detection 
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5) # detect id confidence >0.5

print(f"[INFO] Loading trained model from {config_d.MODEL_PATH}...")
try:
    model = load_model(config_d.MODEL_PATH)
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[CRITICAL ERROR] Could not load model. Did you run train_model_d.py? Error: {e}")
    exit()

# --- 2. START CAMERA ---
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0) # open camera

if not cap.isOpened():
    print("[ERROR] Could not access the webcam.")
    exit()

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get dimensions
    (h, w) = frame.shape[:2]

    # --- 3. DETECT FACES (MediaPipe) ---
    # Convert BGR to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * w)
            ymin = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Fix coordinates (ensure they stay inside the image)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmin + width)
            ymax = min(h, ymin + height)

            # Extract face ROI (Region of Interest)
            face = frame[ymin:ymax, xmin:xmax]
            
            # Ensure face is valid (not empty)
            if face.size == 0:
                continue

            # --- 4. PREDICT MASK (MobileNet) ---
            # Preprocess exactly like we did in training
            try:
                face_input = cv2.resize(face, (224, 224))
                face_input = tf.keras.utils.img_to_array(face_input)
                face_input = preprocess_input(face_input)
                face_input = np.expand_dims(face_input, axis=0) # Add batch dimension

                # PREDICT
                (mask, withoutMask) = model.predict(face_input, verbose=0)[0]
            except Exception as e:
                print(f"Prediction Error: {e}")
                continue

            # --- 5. DRAW RESULTS ---
            # Determine label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Green for Mask, Red for No Mask
            
            # Display probability
            label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

            # Draw box and text
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Show the final frame
    cv2.imshow("Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_detection.close()