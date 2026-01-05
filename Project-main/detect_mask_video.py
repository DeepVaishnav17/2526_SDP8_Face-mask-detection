import cv2
import numpy as np
import winsound
import threading
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. MULTI-THREADING CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- 2. INITIALIZATION ---
print("[INFO] Loading AI models...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.h5")

# Settings
ALARM_THRESHOLD = 20
CONFIDENCE_MIN = 0.75
alarm_counter = 0
results_history = []

# Start Multi-threaded Stream
vs = VideoStream(src=0).start()
print("[INFO] System Live. Press 'q' to exit.")

# Open/Create Log File for SDP8 Report
log_file = open("mask_violations.csv", "a", newline="")
log_writer = csv.writer(log_file)
# Write header only if file is empty
if log_file.tell() == 0:
    log_writer.writerow(["Timestamp", "Status", "Confidence %"])

# --- 3. MAIN LOOP ---
while True:
    frame = vs.read()
    if frame is None: continue
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    mask_status_in_frame = False

    for i in range(0, detections.shape[2]):
        face_confidence = detections[0, 0, i, 2]

        if face_confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(img_to_array(face))
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(face, verbose=0)[0]
            
            # Smoothing Logic
            results_history.append(mask)
            if len(results_history) > 10: results_history.pop(0)
            smoothed_prob = sum(results_history) / len(results_history)

            if smoothed_prob > CONFIDENCE_MIN:
                label, color = "Mask Detected", (0, 255, 0)
            else:
                label, color = "No Mask / Warning!", (0, 0, 255)
                mask_status_in_frame = True 

            display_label = f"{label}: {smoothed_prob * 100:.1f}%"
            cv2.putText(frame, display_label, (startX, startY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

    # 4. ALARM & LOGGING LOGIC
    if mask_status_in_frame:
        alarm_counter += 1
        if alarm_counter >= ALARM_THRESHOLD:
            winsound.Beep(1000, 400)
            # Log violation for report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_writer.writerow([timestamp, "VIOLATION", f"{smoothed_prob*100:.1f}"])
            log_file.flush() # Ensure data is saved immediately
            alarm_counter = 0 
    else:
        alarm_counter = 0 

    cv2.imshow("Security Feed - Multi-threaded AI Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

# Cleanup
print("[INFO] Closing system...")
log_file.close()
vs.stop()
cv2.destroyAllWindows()