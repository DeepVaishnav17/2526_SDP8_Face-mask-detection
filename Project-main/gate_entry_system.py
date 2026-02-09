from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import csv
import os
import threading
import json
from pathlib import Path

app = Flask(__name__)

# Create directories for storing entry data
SNAPSHOTS_DIR = "entry_snapshots"
ENTRY_LOG_FILE = "entry_log.csv"
DAILY_REPORT_FILE = "daily_report.json"

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Global variables
camera = None
detection_active = False
entry_stats = {
    'total_entries': 0,
    'mask_entries': 0,
    'no_mask_entries': 0,
    'current_status': 'Inactive',
    'recent_entries': []  # Last 10 entries
}
stats_lock = threading.Lock()

# Load models
print("[INFO] Loading AI models for Gate Entry System...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector_best.h5")
print("[INFO] Models loaded successfully!")

# Detection settings
CONFIDENCE_MIN = 0.75
ENTRY_COOLDOWN = 5  # seconds between same person entries

# Tracking data
tracked_persons = {}  # {person_id: {'last_entry_time', 'mask_status', 'entry_count'}}
person_id_counter = 0
frame_counter = 0

class PersonTracker:
    """Track persons entering the building"""
    def __init__(self):
        self.tracked = {}
        self.next_id = 1
        self.position_threshold = 100  # pixels
        
    def get_person_id(self, box, current_time):
        """Get or create person ID based on position"""
        (x1, y1, x2, y2) = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Check if person already exists nearby
        for pid, data in list(self.tracked.items()):
            if 'position' not in data:
                continue
                
            px, py = data['position']
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            
            if dist < self.position_threshold:
                # Update position
                self.tracked[pid]['position'] = (cx, cy)
                self.tracked[pid]['last_seen'] = current_time
                return pid
        
        # New person detected
        new_id = self.next_id
        self.next_id += 1
        self.tracked[new_id] = {
            'position': (cx, cy),
            'last_seen': current_time,
            'entered': False
        }
        return new_id
    
    def cleanup_old_tracks(self, current_time, timeout=5):
        """Remove persons not seen for timeout seconds"""
        to_remove = []
        for pid, data in self.tracked.items():
            if (current_time - data['last_seen']).total_seconds() > timeout:
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.tracked[pid]

tracker = PersonTracker()
results_history = {}  # {person_id: [mask_probs]}

def save_entry_snapshot(frame, person_id, mask_status, confidence):
    """Save snapshot of person entering"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"entry_{timestamp}_ID{person_id}_{mask_status}.jpg"
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    
    # Add entry info overlay
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    cv2.putText(frame, f"ENTRY LOG", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Person ID: {person_id}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Mask: {mask_status} ({confidence:.1f}%)", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if mask_status == "DETECTED" else (0, 0, 255), 2)
    
    cv2.imwrite(filepath, frame)
    return filename

def log_entry(person_id, mask_status, confidence, snapshot_file):
    """Log entry to CSV file"""
    try:
        file_exists = os.path.isfile(ENTRY_LOG_FILE)
        with open(ENTRY_LOG_FILE, "a", newline="") as log_file:
            log_writer = csv.writer(log_file)
            if not file_exists:
                log_writer.writerow(["Timestamp", "Person_ID", "Mask_Status", 
                                    "Confidence_%", "Snapshot_File"])
            log_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                person_id,
                mask_status,
                f"{confidence:.2f}",
                snapshot_file
            ])
    except Exception as e:
        print(f"[ERROR] Entry logging failed: {e}")

def update_recent_entries(person_id, mask_status, confidence, snapshot_file):
    """Update recent entries list"""
    with stats_lock:
        entry_data = {
            'id': person_id,
            'time': datetime.now().strftime("%H:%M:%S"),
            'status': mask_status,
            'confidence': f"{confidence:.1f}%",
            'snapshot': snapshot_file
        }
        entry_stats['recent_entries'].insert(0, entry_data)
        # Keep only last 10 entries
        entry_stats['recent_entries'] = entry_stats['recent_entries'][:10]

def detect_and_predict_mask(frame):
    """Detect faces and predict mask status for entry tracking"""
    global results_history, entry_stats, tracker, frame_counter
    
    frame_counter += 1
    current_time = datetime.now()
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Draw entry detection zone (virtual gate)
    zone_y = int(h * 0.6)  # Entry line at 60% of frame height
    cv2.line(frame, (0, zone_y), (w, zone_y), (0, 255, 255), 2)
    cv2.putText(frame, "ENTRY ZONE", (10, zone_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    active_persons = []
    
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
            
            # Get person ID
            person_id = tracker.get_person_id((startX, startY, endX, endY), current_time)
            active_persons.append(person_id)
                
            # Predict mask
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = preprocess_input(img_to_array(face_rgb))
            face_rgb = np.expand_dims(face_rgb, axis=0)

            (mask, withoutMask) = maskNet.predict(face_rgb, verbose=0)[0]
            
            # Smoothing
            if person_id not in results_history:
                results_history[person_id] = []
            results_history[person_id].append(mask)
            if len(results_history[person_id]) > 10:
                results_history[person_id].pop(0)
            smoothed_prob = sum(results_history[person_id]) / len(results_history[person_id])

            if smoothed_prob > CONFIDENCE_MIN:
                label = "MASK"
                mask_status = "DETECTED"
                color = (0, 255, 0)
            else:
                label = "NO MASK"
                mask_status = "NOT_DETECTED"
                color = (0, 0, 255)

            # Check if person is crossing entry zone
            face_center_y = (startY + endY) // 2
            
            if not tracker.tracked[person_id]['entered']:
                # Person crossing the entry line
                if face_center_y > zone_y - 50 and face_center_y < zone_y + 50:
                    tracker.tracked[person_id]['entered'] = True
                    
                    # Log the entry
                    snapshot_file = save_entry_snapshot(frame, person_id, mask_status, 
                                                       smoothed_prob * 100)
                    log_entry(person_id, mask_status, smoothed_prob * 100, snapshot_file)
                    update_recent_entries(person_id, mask_status, smoothed_prob * 100, 
                                         snapshot_file)
                    
                    with stats_lock:
                        entry_stats['total_entries'] += 1
                        if mask_status == "DETECTED":
                            entry_stats['mask_entries'] += 1
                        else:
                            entry_stats['no_mask_entries'] += 1
                    
                    print(f"[ENTRY] Person {person_id} - {mask_status} - {smoothed_prob*100:.1f}%")

            # Draw detection box
            display_label = f"ID{person_id}: {label} {smoothed_prob*100:.0f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            
            # Label background
            label_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (startX, startY-35), (startX+label_size[0]+10, startY), color, -1)
            cv2.putText(frame, display_label, (startX+5, startY-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Entry status indicator
            if tracker.tracked[person_id]['entered']:
                cv2.putText(frame, "LOGGED", (startX, endY+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Cleanup old tracks
    tracker.cleanup_old_tracks(current_time)
    
    # Remove old history
    for pid in list(results_history.keys()):
        if pid not in active_persons:
            del results_history[pid]
    
    # Update status
    with stats_lock:
        if len(active_persons) > 0:
            entry_stats['current_status'] = f"{len(active_persons)} Person(s) Detected"
        else:
            entry_stats['current_status'] = "Monitoring..."

    # Display stats overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (w-5, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    with stats_lock:
        cv2.putText(frame, f"GATE ENTRY MONITORING SYSTEM", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Entries Today: {entry_stats['total_entries']} | " +
                          f"Mask: {entry_stats['mask_entries']} | " +
                          f"No Mask: {entry_stats['no_mask_entries']}", 
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame

def generate_frames():
    """Generate frames for video streaming"""
    global camera, detection_active
    
    if camera is not None:
        try:
            camera.release()
        except:
            pass
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    import time
    time.sleep(0.5)
    
    try:
        while detection_active:
            if camera is None or not camera.isOpened():
                break
                
            success, frame = camera.read()
            if not success:
                break
            
            frame = detect_and_predict_mask(frame)
            
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
    return render_template('gate_entry.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detection_active, results_history, tracker
    detection_active = True
    results_history = {}
    tracker = PersonTracker()
    return jsonify({'status': 'started'})

@app.route('/stop_detection')
def stop_detection():
    global detection_active, camera
    detection_active = False
    
    import time
    time.sleep(0.3)
    
    if camera is not None:
        try:
            camera.release()
            camera = None
        except Exception as e:
            print(f"[WARNING] Error releasing camera: {e}")
    
    return jsonify({'status': 'stopped'})

@app.route('/get_stats')
def get_stats():
    with stats_lock:
        return jsonify(entry_stats)

@app.route('/reset_stats')
def reset_stats():
    global entry_stats, results_history, tracker
    with stats_lock:
        entry_stats = {
            'total_entries': 0,
            'mask_entries': 0,
            'no_mask_entries': 0,
            'current_status': 'Inactive',
            'recent_entries': []
        }
    results_history = {}
    tracker = PersonTracker()
    return jsonify({'status': 'reset'})

@app.route('/snapshots/<path:filename>')
def get_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)

@app.route('/download_log')
def download_log():
    return send_from_directory('.', ENTRY_LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏢  GATE ENTRY MONITORING SYSTEM - MASK DETECTION")
    print("="*70)
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📊 Entry logs saved to:", ENTRY_LOG_FILE)
    print("📸 Snapshots saved to:", SNAPSHOTS_DIR)
    print("="*70 + "\n")
    app.run(debug=True, threaded=True, port=5000)
