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
maskNet = load_model("mask_detector_best.h5")  # Using improved model with 98.25% accuracy
print("[INFO] Models loaded successfully!")

# Settings
CONFIDENCE_MIN = 0.75
results_history = {}  # {face_id: [mask_probs]}
previous_state = {}  # {face_id: 'mask'/'no_mask'}
face_positions = {}  # {face_id: (x, y, w, h)}

def get_face_id(box):
    """Get face ID based on position"""
    global face_positions
    (x1, y1, x2, y2) = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    
    for fid, (px1, py1, px2, py2) in face_positions.items():
        pcx, pcy = (px1+px2)//2, (py1+py2)//2
        dist = ((cx-pcx)**2 + (cy-pcy)**2)**0.5
        if dist < 80:  # Same face if center within 80px
            return fid
    return max(face_positions.keys(), default=-1) + 1

def detect_and_predict_mask(frame):
    """Detect faces and predict mask status"""
    global results_history, stats, previous_state, face_positions
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    current_faces = {}
    mask_count = 0
    no_mask_count = 0

    for i in range(0, detections.shape[2]):
        face_confidence = detections[0, 0, i, 2]

        if face_confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            
            # Get unique face ID
            face_id = get_face_id((startX, startY, endX, endY))
            current_faces[face_id] = (startX, startY, endX, endY)
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(img_to_array(face))
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(face, verbose=0)[0]
            
            # Per-face smoothing
            if face_id not in results_history:
                results_history[face_id] = []
            results_history[face_id].append(mask)
            if len(results_history[face_id]) > 10:
                results_history[face_id].pop(0)
            smoothed_prob = sum(results_history[face_id]) / len(results_history[face_id])

            if smoothed_prob > CONFIDENCE_MIN:
                label, color = "Mask", (0, 255, 0)
                current_state = 'mask'
                mask_count += 1
            else:
                label, color = "No Mask", (0, 0, 255)
                current_state = 'no_mask'
                no_mask_count += 1

            # Track state changes per face
            if face_id not in previous_state:
                previous_state[face_id] = None
            
            if previous_state[face_id] != current_state:
                with stats_lock:
                    stats['total_detections'] += 1
                    if current_state == 'mask':
                        stats['mask_count'] += 1
                    else:
                        stats['no_mask_count'] += 1
                        log_violation(smoothed_prob)
                previous_state[face_id] = current_state

            display_label = f"ID{face_id}: {label} {smoothed_prob*100:.0f}%"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (startX, startY-30), (startX+label_size[0], startY), color, -1)
            cv2.putText(frame, display_label, (startX, startY-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Update tracking
    face_positions = current_faces
    for fid in list(results_history.keys()):
        if fid not in current_faces:
            del results_history[fid]
            del previous_state[fid]
    
    # Update status
    with stats_lock:
        if no_mask_count > 0:
            stats['current_status'] = f"{no_mask_count} No Mask Warning!"
        elif mask_count > 0:
            stats['current_status'] = f"{mask_count} Mask Detected"
        else:
            stats['current_status'] = "No Faces"

    # Add info overlay
    info_text = f"Faces: {len(current_faces)} | Mask: {mask_count} | No Mask: {no_mask_count}"
    cv2.rectangle(frame, (5, 5), (w-5, 50), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
    
    # Optimize camera settings for better performance
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
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
    global detection_active, results_history, previous_state, face_positions
    detection_active = True
    results_history = {}
    previous_state = {}
    face_positions = {}
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
    global stats, results_history, previous_state, face_positions
    with stats_lock:
        stats = {
            'total_detections': 0,
            'mask_count': 0,
            'no_mask_count': 0,
            'current_status': 'Inactive'
        }
    results_history = {}
    previous_state = {}
    face_positions = {}
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  FACE MASK DETECTION SYSTEM")
    print("="*60)
    print("📱 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, threaded=True, port=5000)
