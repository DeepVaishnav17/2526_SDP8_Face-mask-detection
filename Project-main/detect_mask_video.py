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

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32, verbose=0)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# --- 2. INITIALIZATION ---
print("[INFO] Loading AI models...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.h5")

# Settings
ALARM_THRESHOLD = 20
CONFIDENCE_MIN = 0.75
alarm_counter = 0
# results_history = []

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
    
    # 1. Detect faces and predict masks (BATCH PROCESSING)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    mask_status_in_frame = False # Reset status for this frame

    # 2. Loop over the detected face locations and their corresponding predictions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine class label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # If ANY person has no mask, set the flag to trigger alarm later
        if label == "No Mask":
            mask_status_in_frame = True

        # Display label and probability
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # 4. ALARM & LOGGING LOGIC
    # (This logic remains largely the same, but now it triggers if *anyone* in the frame has no mask)
    if mask_status_in_frame:
        alarm_counter += 1
        if alarm_counter >= ALARM_THRESHOLD:
            winsound.Beep(1000, 400)
            # Log violation for report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Note: We log the violation, but we don't have a specific probability 
            # for the CSV since there might be multiple people. 
            # You can log "Multiple" or just the current timestamp.
            log_writer.writerow([timestamp, "VIOLATION", "N/A"]) 
            log_file.flush()
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