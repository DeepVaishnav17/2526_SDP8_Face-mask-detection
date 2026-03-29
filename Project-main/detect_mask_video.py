
# import cv2
# import numpy as np
# import winsound
# import threading
# import os
# import collections
# from datetime import datetime
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # --- 1. SETUP & CONFIGURATION ---
# AUDIT_DIR = "accuracy_audit"
# os.makedirs(f"{AUDIT_DIR}/mask", exist_ok=True)
# os.makedirs(f"{AUDIT_DIR}/no_mask", exist_ok=True)

# # Performance Tuning for Production
# STABILITY_WINDOW = 10     # Consensus required over the last 10 frames
# MIN_SEEN_FRAMES = 5       # Ignore "ghost" detections seen for < 5 frames
# CONFIDENCE_THRESHOLD = 0.90 # High precision requirement to reduce False Positives

# # --- 2. MULTI-THREADING CLASS ---
# class VideoStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         threading.Thread(target=self.update, args=(), daemon=True).start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped: return
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self): return self.frame

#     def stop(self): self.stopped = True

# # --- 3. ADVANCED PREPROCESSING ---
# def apply_clahe(img):
#     """Normalize lighting on face crops using Contrast Limited Adaptive Histogram Equalization."""
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     lab[:,:,0] = clahe.apply(lab[:,:,0])
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# # --- 4. INITIALIZATION ---
# print("[INFO] Initializing Perfected Production Environment...")
# faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# maskNet = load_model("mask_detector.h5")

# # Tracking & Jury Dictionaries
# face_tracks = {} # {id: {'centroid': (x,y), 'seen_count': int}}
# face_queues = collections.defaultdict(lambda: collections.deque(maxlen=STABILITY_WINDOW))
# next_id = 0
# audit_counter = 0

# vs = VideoStream(src=0).start()

# # --- 5. MAIN LOOP ---
# while True:
#     frame = vs.read()
#     if frame is None: continue
    
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     faceNet.setInput(blob)
#     detections = faceNet.forward()

#     faces, locs, centroids = [], [], []

#     # Phase 1: Robust Detection & Preprocessing
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.65:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             (startX, startY) = (max(0, startX), max(0, startY))
#             (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#             face_crop = frame[startY:endY, startX:endX]
#             if face_crop.size > 0:
#                 centroids.append(((startX + endX) // 2, (startY + endY) // 2))
#                 locs.append((startX, startY, endX, endY))
                
#                 # Standardization for Overfitted Models
#                 face_norm = apply_clahe(face_crop)
#                 face_norm = cv2.resize(face_norm, (224, 224))
#                 faces.append(preprocess_input(img_to_array(face_norm)))

#     # Phase 2: Jury Inference & Centroid Tracking
#     if len(faces) > 0:
#         preds = maskNet.predict(np.array(faces), batch_size=32, verbose=0)
#         current_frame_ids = []

#         for (box, pred, centroid) in zip(locs, preds, centroids):
#             (startX, startY, endX, endY) = box
#             (mask, noMask) = pred

#             # 1. Centroid Tracking to maintain identity
#             matched_id = None
#             for fid, data in face_tracks.items():
#                 dist = np.linalg.norm(np.array(centroid) - np.array(data['centroid']))
#                 if dist < 65: 
#                     matched_id = fid
#                     break
            
#             if matched_id is None:
#                 matched_id = next_id
#                 face_tracks[matched_id] = {'centroid': centroid, 'seen_count': 1}
#                 next_id += 1
#             else:
#                 face_tracks[matched_id]['centroid'] = centroid
#                 face_tracks[matched_id]['seen_count'] += 1
            
#             current_frame_ids.append(matched_id)
#             face_queues[matched_id].append(mask)
            
#             # 2. THE JURY SYSTEM: Only display labels for stable detections
#             if face_tracks[matched_id]['seen_count'] >= MIN_SEEN_FRAMES:
#                 # Average the last 10 frames for this specific person
#                 smoothed_prob = sum(face_queues[matched_id]) / len(face_queues[matched_id])

#                 label = "Mask" if smoothed_prob > CONFIDENCE_THRESHOLD else "NO MASK"
#                 color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

#                 # 3. Audit Capture for Manual Accuracy Verification
#                 audit_counter += 1
#                 if audit_counter % 50 == 0:
#                     folder = "mask" if label == "Mask" else "no_mask"
#                     fname = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
#                     cv2.imwrite(f"{AUDIT_DIR}/{folder}/{fname}", frame[startY:endY, startX:endX])

#                 cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
#                 cv2.putText(frame, f"ID {matched_id}: {label} ({smoothed_prob*100:.0f}%)", (startX, startY-10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

#         # Cleanup lost tracks
#         face_tracks = {fid: d for fid, d in face_tracks.items() if fid in current_frame_ids}
#         for fid in list(face_queues.keys()):
#             if fid not in current_frame_ids: del face_queues[fid]

#     cv2.imshow("Perfected Mask Tracker (CLAHE + Jury System)", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"): break

# vs.stop()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import winsound
# import threading
# import csv
# from datetime import datetime
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # --- 1. MULTI-THREADING CLASS (Unchanged) ---
# class VideoStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         threading.Thread(target=self.update, args=(), daemon=True).start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped: return
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # --- 2. INITIALIZATION ---
# print("[INFO] Loading AI models...")
# faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# maskNet = load_model("mask_detector.h5")

# # Production Settings
# ALARM_THRESHOLD = 15 
# CONFIDENCE_MIN = 0.80 # Slightly higher for production
# face_histories = {} # Dictionary to store individual face scores
# alarm_counter = 0

# vs = VideoStream(src=0).start()
# log_file = open("mask_violations.csv", "a", newline="")
# log_writer = csv.writer(log_file)
# if log_file.tell() == 0:
#     log_writer.writerow(["Timestamp", "Status", "Confidence %"])

# # --- 3. MAIN LOOP ---
# while True:
#     frame = vs.read()
#     if frame is None: continue
    
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     faceNet.setInput(blob)
#     detections = faceNet.forward()

#     faces = []
#     locs = []
#     preds = []

#     # Phase 1: Detect Faces and Prepare Crops
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5: # Face detection confidence
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             (startX, startY) = (max(0, startX), max(0, startY))
#             (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#             face = frame[startY:endY, startX:endX]
#             if face.size > 0:
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                 face = cv2.resize(face, (224, 224))
#                 face = img_to_array(face)
#                 face = preprocess_input(face)

#                 faces.append(face)
#                 locs.append((startX, startY, endX, endY))

#     # Phase 2: Batch Prediction (Faster for multiple people)
#     if len(faces) > 0:
#         faces = np.array(faces, dtype="float32")
#         preds = maskNet.predict(faces, batch_size=32, verbose=0)

#     # Phase 3: Individual Tracking and Drawing
#     mask_status_in_frame = False
    
#     for (box, pred) in zip(locs, preds):
#         (startX, startY, endX, endY) = box
#         (mask, withoutMask) = pred

#         # Create a unique ID for this face based on position (grid-based tracking)
#         face_id = (startX // 50, startY // 50)
        
#         # Individual Smoothing Logic (EMA)
#         if face_id not in face_histories:
#             face_histories[face_id] = mask
#         else:
#             # Exponential Moving Average: 70% history, 30% current frame
#             face_histories[face_id] = (face_histories[face_id] * 0.7) + (mask * 0.3)

#         smoothed_prob = face_histories[face_id]

#         if smoothed_prob > CONFIDENCE_MIN:
#             label, color = "Mask Detected", (0, 255, 0)
#         else:
#             label, color = "No Mask / Warning!", (0, 0, 255)
#             mask_status_in_frame = True 

#         display_label = f"{label}: {smoothed_prob * 100:.1f}%"
#         cv2.putText(frame, display_label, (startX, startY - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

#     # Clean up old face IDs to prevent memory leaks
#     if len(face_histories) > 20: face_histories.clear()

#     # 4. ALARM LOGIC
#     if mask_status_in_frame:
#         alarm_counter += 1
#         if alarm_counter >= ALARM_THRESHOLD:
#             winsound.Beep(1000, 400)
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_writer.writerow([timestamp, "VIOLATION", "N/A"])
#             log_writer.writerow([timestamp, "VIOLATION", f"Threshold reached"])
#             log_file.flush()
#             alarm_counter = 0 
#     else:
#         alarm_counter = 0 

#     cv2.imshow("Production Security Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"): break

# log_file.close()
# vs.stop()
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import winsound
# import threading
# import csv
# from datetime import datetime
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # --- 1. MULTI-THREADING CLASS (Unchanged) ---
# class VideoStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()
#         self.stopped = False

#     def start(self):
#         threading.Thread(target=self.update, args=(), daemon=True).start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped: return
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self):
#         return self.frame

#     def stop(self):
#         self.stopped = True

# # --- 2. INITIALIZATION ---
# print("[INFO] Loading AI models...")
# faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# maskNet = load_model("mask_detector.h5")

# # Production Settings
# ALARM_THRESHOLD = 15 
# CONFIDENCE_MIN = 0.80 # Slightly higher for production
# face_histories = {} # Dictionary to store individual face scores
# alarm_counter = 0

# vs = VideoStream(src=0).start()
# log_file = open("mask_violations.csv", "a", newline="")
# log_writer = csv.writer(log_file)
# if log_file.tell() == 0:
#     log_writer.writerow(["Timestamp", "Status", "Confidence %"])

# # --- 3. MAIN LOOP ---
# while True:
#     frame = vs.read()
#     if frame is None: continue
    
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     faceNet.setInput(blob)
#     detections = faceNet.forward()

#     faces = []
#     locs = []
#     preds = []

#     # Phase 1: Detect Faces and Prepare Crops
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5: # Face detection confidence
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             (startX, startY) = (max(0, startX), max(0, startY))
#             (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#             face = frame[startY:endY, startX:endX]
#             if face.size > 0:
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                 face = cv2.resize(face, (224, 224))
#                 face = img_to_array(face)
#                 face = preprocess_input(face)

#                 faces.append(face)
#                 locs.append((startX, startY, endX, endY))

#     # Phase 2: Batch Prediction (Faster for multiple people)
#     if len(faces) > 0:
#         faces = np.array(faces, dtype="float32")
#         preds = maskNet.predict(faces, batch_size=32, verbose=0)

#     # Phase 3: Individual Tracking and Drawing
#     mask_status_in_frame = False
    
#     for (box, pred) in zip(locs, preds):
#         (startX, startY, endX, endY) = box
#         (mask, withoutMask) = pred

#         # Create a unique ID for this face based on position (grid-based tracking)
#         face_id = (startX // 50, startY // 50)
        
#         # Individual Smoothing Logic (EMA)
#         if face_id not in face_histories:
#             face_histories[face_id] = mask
#         else:
#             # Exponential Moving Average: 70% history, 30% current frame
#             face_histories[face_id] = (face_histories[face_id] * 0.7) + (mask * 0.3)

#         smoothed_prob = face_histories[face_id]

#         if smoothed_prob > CONFIDENCE_MIN:
#             label, color = "Mask Detected", (0, 255, 0)
#         else:
#             label, color = "No Mask / Warning!", (0, 0, 255)
#             mask_status_in_frame = True 

#         display_label = f"{label}: {smoothed_prob * 100:.1f}%"
#         cv2.putText(frame, display_label, (startX, startY - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

#     # Clean up old face IDs to prevent memory leaks
#     if len(face_histories) > 20: face_histories.clear()

#     # 4. ALARM LOGIC
#     if mask_status_in_frame:
#         alarm_counter += 1
#         if alarm_counter >= ALARM_THRESHOLD:
#             winsound.Beep(1000, 400)
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_writer.writerow([timestamp, "VIOLATION", "N/A"])
#             log_writer.writerow([timestamp, "VIOLATION", f"Threshold reached"])
#             log_file.flush()
#             alarm_counter = 0 
#     else:
#         alarm_counter = 0 

#     cv2.imshow("Production Security Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"): break

# log_file.close()
# vs.stop()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import winsound
import threading
import os
import collections
import requests
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[WARNING] 'face_recognition' library not found. Face recognition disabled.")

import pickle

# --- 1. SETUP & CONFIGURATION ---
API_URL = "http://localhost:5000"
LAST_SENT_TIME = 0
SEND_INTERVAL = 2.0  # Seconds between DB updates to avoid spam

AUDIT_DIR = "accuracy_audit"
os.makedirs(f"{AUDIT_DIR}/mask", exist_ok=True)
os.makedirs(f"{AUDIT_DIR}/no_mask", exist_ok=True)

# Load Face Encodings
data = {"encodings": [], "names": []}
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    print(f"[INFO] Loaded {len(data['encodings'])} face encodings.")
except:
    print("[WARNING] No face encodings found. Face recognition will be disabled.")


# Performance Tuning for Production
STABILITY_WINDOW = 10     # Consensus required over the last 10 frames
MIN_SEEN_FRAMES = 5       # Ignore "ghost" detections seen for < 5 frames
CONFIDENCE_THRESHOLD = 0.90 # High precision requirement to reduce False Positives

# --- 2. MULTI-THREADING CLASS ---
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

    def read(self): return self.frame

    def stop(self): self.stopped = True

# --- 3. ADVANCED PREPROCESSING ---
def apply_clahe(img):
    """Normalize lighting on face crops using Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# --- 4. INITIALIZATION ---
print("[INFO] Initializing Perfected Production Environment...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.h5")

# Tracking & Jury Dictionaries
face_tracks = {} # {id: {'centroid': (x,y), 'seen_count': int}}
face_queues = collections.defaultdict(lambda: collections.deque(maxlen=STABILITY_WINDOW))
next_id = 0
audit_counter = 0

vs = VideoStream(src=0).start()

# --- 5. MAIN LOOP ---
while True:
    frame = vs.read()
    if frame is None: continue
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, centroids = [], [], []

    # Phase 1: Robust Detection & Preprocessing
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.65:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face_crop = frame[startY:endY, startX:endX]
            if face_crop.size > 0:
                centroids.append(((startX + endX) // 2, (startY + endY) // 2))
                locs.append((startX, startY, endX, endY))
                
                # Standardization for Overfitted Models
                face_norm = apply_clahe(face_crop)
                face_norm = cv2.resize(face_norm, (224, 224))
                faces.append(preprocess_input(img_to_array(face_norm)))

    # Phase 2: Jury Inference & Centroid Tracking
    if len(faces) > 0:
        preds = maskNet.predict(np.array(faces), batch_size=32, verbose=0)
        current_frame_ids = []

        for (box, pred, centroid) in zip(locs, preds, centroids):
            (startX, startY, endX, endY) = box
            (mask, noMask) = pred

            # --- FACE RECOGNITION (Optimized) ---
            name = "Unknown"
            if FACE_REC_AVAILABLE and len(data["encodings"]) > 0:
                # Convert face coordinates for face_recognition [top, right, bottom, left]
                # We already have the crops, but let's re-encode from the full frame for accuracy
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pass the specific box we already found to avoid re-detection
                # face_recognition expects (top, right, bottom, left)
                face_box = [(startY, endX, endY, startX)]
                
                encodings = face_recognition.face_encodings(rgb_small, face_box)
                
                if len(encodings) > 0:
                    encoding = encodings[0]
                    matches = face_recognition.compare_faces(data["encodings"], encoding)
                    
                    if True in matches:
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)

            # 1. Centroid Tracking to maintain identity
            matched_id = None
            for fid, data_track in face_tracks.items(): # Renamed 'data' to 'data_track' to avoid conflict
                dist = np.linalg.norm(np.array(centroid) - np.array(data_track['centroid']))
                if dist < 65: 
                    matched_id = fid
                    break
            
            if matched_id is None:
                matched_id = next_id
                face_tracks[matched_id] = {'centroid': centroid, 'seen_count': 1, 'name': name}
                next_id += 1
            else:
                face_tracks[matched_id]['centroid'] = centroid
                face_tracks[matched_id]['seen_count'] += 1
                # Update name if we found a known one (and previous was Unknown)
                if name != "Unknown":
                    face_tracks[matched_id]['name'] = name
            
            current_frame_ids.append(matched_id)
            face_queues[matched_id].append(mask)
            
            # Use the tracked name
            display_name = face_tracks[matched_id].get('name', 'Unknown')

            # 2. THE JURY SYSTEM: Only display labels for stable detections
            if face_tracks[matched_id]['seen_count'] >= MIN_SEEN_FRAMES:
                # Average the last 10 frames for this specific person
                smoothed_prob = sum(face_queues[matched_id]) / len(face_queues[matched_id])

                label = "Mask" if smoothed_prob > CONFIDENCE_THRESHOLD else "NO MASK"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # 3. Audit Capture for Manual Accuracy Verification
                audit_counter += 1
                if audit_counter % 50 == 0:
                    folder = "mask" if label == "Mask" else "no_mask"
                    fname = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    cv2.imwrite(f"{AUDIT_DIR}/{folder}/{fname}", frame[startY:endY, startX:endX])

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, f"{display_name}: {label} ({smoothed_prob*100:.0f}%)", (startX, startY-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                # --- NEW: History Tracking ---
                # Check directly in the loop (simple approach)
                current_time = time.time()
                if current_time - LAST_SENT_TIME > SEND_INTERVAL:
                    try:
                        # Only record if we actually know the user? Or record "Unknown" too?
                        # Let's record if name is known, OR if we fall back to API global user
                        
                        target_user = display_name
                        if target_user == "Unknown":
                             # Fallback to manual endpoint
                            r = requests.get(f"{API_URL}/get_user", timeout=0.1)
                            if r.status_code == 200:
                                manual_user = r.json().get("current_user")
                                if manual_user:
                                    target_user = manual_user
                        
                        if target_user != "Unknown":
                            # 2. Send Record
                            requests.post(f"{API_URL}/record_status", json={
                                "name": target_user,
                                "status": label
                            }, timeout=0.1)
                            LAST_SENT_TIME = current_time
                    except Exception as e:
                        pass # Fail silently to not crash video loop

        # Cleanup lost tracks
        face_tracks = {fid: d for fid, d in face_tracks.items() if fid in current_frame_ids}
        for fid in list(face_queues.keys()):
            if fid not in current_frame_ids: del face_queues[fid]

    cv2.imshow("Perfected Mask Tracker (CLAHE + Jury System)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

vs.stop()
cv2.destroyAllWindows()