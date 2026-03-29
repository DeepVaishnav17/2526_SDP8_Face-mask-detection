from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import csv
import os
import threading

app = Flask(__name__)

# Global variables
camera = None
detection_active = False
stats = {
    'total_detections': 0,
    'mask_count': 0,
    'no_mask_count': 0,
    'current_status': 'Inactive'
}
stats_lock = threading.Lock()

# Load models
print("[INFO] Loading AI models...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.h5")
print("[INFO] Models loaded successfully!")

# Settings
CONFIDENCE_MIN = 0.75
results_history = []
previous_state = None  # Track previous detection state for session-based counting

def detect_and_predict_mask(frame):
    """Detect faces and predict mask status"""
    global results_history, stats, previous_state
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces_detected = 0
    mask_detected = False

    for i in range(0, detections.shape[2]):
        face_confidence = detections[0, 0, i, 2]

        if face_confidence > 0.5:
            faces_detected += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(img_to_array(face))
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(face, verbose=0)[0]
            
            # Smoothing Logic
            results_history.append(mask)
            if len(results_history) > 10:
                results_history.pop(0)
            smoothed_prob = sum(results_history) / len(results_history)

            if smoothed_prob > CONFIDENCE_MIN:
                label, color = "Mask Detected", (0, 255, 0)
                mask_detected = True
                current_state = 'mask'
            else:
                label, color = "No Mask - Warning!", (0, 0, 255)
                current_state = 'no_mask'

            # Only increment counters when state CHANGES (session-based counting)
            if previous_state != current_state:
                with stats_lock:
                    stats['total_detections'] += 1
                    if current_state == 'mask':
                        stats['mask_count'] += 1
                    else:
                        stats['no_mask_count'] += 1
                        # Log violation only on state change
                        log_violation(smoothed_prob)
                
                previous_state = current_state

            display_label = f"{label}: {smoothed_prob * 100:.1f}%"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            cv2.putText(frame, display_label, (startX, startY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            with stats_lock:
                stats['current_status'] = label

    # Add info overlay
    info_text = f"Faces Detected: {faces_detected} | Time: {datetime.now().strftime('%H:%M:%S')}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def log_violation(confidence):
    """Log mask violations to CSV file"""
    try:
        file_exists = os.path.isfile("mask_violations.csv")
        with open("mask_violations.csv", "a", newline="") as log_file:
            log_writer = csv.writer(log_file)
            if not file_exists:
                log_writer.writerow(["Timestamp", "Status", "Confidence %"])
            log_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "No Mask",
                f"{confidence * 100:.2f}"
            ])
    except Exception as e:
        print(f"[ERROR] Logging failed: {e}")

def generate_frames():
    """Generate frames for video streaming"""
    global camera, detection_active
    
    # Release any existing camera before opening new one
    if camera is not None:
        try:
            camera.release()
        except:
            pass
    
    camera = cv2.VideoCapture(0)
    
    # Give camera time to initialize
    import time
    time.sleep(0.5)
    
    try:
        while detection_active:
            if camera is None or not camera.isOpened():
                break
                
            success, frame = camera.read()
            if not success:
                break
            
            # Process frame
            frame = detect_and_predict_mask(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        if camera is not None:
            camera.release()
            camera = None

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    """Start detection"""
    global detection_active, results_history, previous_state
    detection_active = True
    results_history = []
    previous_state = None  # Reset state when starting new session
    return jsonify({'status': 'started'})

@app.route('/stop_detection')
def stop_detection():
    """Stop detection"""
    global detection_active, camera
    detection_active = False
    
    # Give generator time to exit gracefully
    import time
    time.sleep(0.3)
    
    # Release camera
    if camera is not None:
        try:
            camera.release()
            camera = None
        except Exception as e:
            print(f"[WARNING] Error releasing camera: {e}")
    
    return jsonify({'status': 'stopped'})

@app.route('/get_stats')
def get_stats():
    """Get current statistics"""
    with stats_lock:
        return jsonify(stats)

@app.route('/reset_stats')
def reset_stats():
    """Reset statistics"""
    global stats
    with stats_lock:
        stats = {
            'total_detections': 0,
            'mask_count': 0,
            'no_mask_count': 0,
            'current_status': 'Inactive'
        }
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  FACE MASK DETECTION SYSTEM")
    print("="*60)
    print("📱 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, threaded=True, port=5000)
