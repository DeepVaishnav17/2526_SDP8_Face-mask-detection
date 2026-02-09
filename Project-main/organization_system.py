from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import os
import threading
import base64
from database import OrganizationDatabase
from face_recognizer import SimpleFaceRecognizer

app = Flask(__name__)

# Create directories
REGISTRATION_PHOTOS_DIR = "registration_photos"
ENTRY_SNAPSHOTS_DIR = "entry_snapshots"
os.makedirs(REGISTRATION_PHOTOS_DIR, exist_ok=True)
os.makedirs(ENTRY_SNAPSHOTS_DIR, exist_ok=True)

# Initialize systems
print("[INFO] Initializing Organization Gate Entry System...")
db = OrganizationDatabase()
face_recognizer = SimpleFaceRecognizer()

# Load mask detection models
print("[INFO] Loading mask detection models...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector_best.h5")
print("[INFO] All models loaded successfully!")

# Global variables
camera = None
detection_active = False
stats_lock = threading.Lock()
entry_cooldown = {}  # {person_id: last_entry_time}
COOLDOWN_SECONDS = 10

def save_entry_snapshot(frame, person_name, person_id, mask_status, confidence):
    """Save snapshot of person entering"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"entry_{timestamp}_{person_name.replace(' ', '_')}_{mask_status}.jpg"
    filepath = os.path.join(ENTRY_SNAPSHOTS_DIR, filename)
    
    # Add entry info overlay
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, f"ORGANIZATION ENTRY LOG", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Name: {person_name}", (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"ID: {person_id}", (20, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Mask: {mask_status} ({confidence:.1f}%)", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if mask_status == "DETECTED" else (0, 0, 255), 2)
    
    cv2.imwrite(filepath, frame)
    return filename

def can_log_entry(person_id):
    """Check if enough time has passed since last entry"""
    now = datetime.now()
    if person_id in entry_cooldown:
        time_diff = (now - entry_cooldown[person_id]).total_seconds()
        if time_diff < COOLDOWN_SECONDS:
            return False
    return True

def detect_and_predict_mask(frame):
    """Detect faces, recognize persons, and predict mask status"""
    global entry_cooldown
    
    (h, w) = frame.shape[:2]
    
    # Draw entry zone
    zone_y = int(h * 0.6)
    cv2.line(frame, (0, zone_y), (w, zone_y), (0, 255, 255), 3)
    cv2.putText(frame, "ENTRY ZONE", (10, zone_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    entries_this_frame = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            if endX - startX < 50 or endY - startY < 50:
                continue
            
            # Extract face for recognition
            face_region = frame[startY:endY, startX:endX]
            if face_region.size == 0:
                continue
            
            # Recognize person
            face_label, person_name, recog_confidence = face_recognizer.recognize_face(
                frame, confidence_threshold=5  # Very low threshold for ORB matching
            )
            
            # Debug what face recognizer returned
            print(f"[FR_RESULT] Label: {face_label}, Name: {person_name}, Conf: {recog_confidence}%")
            
            # Get person from database using face label
            db_person = None
            if face_label is not None:
                db_person = db.get_person_by_face_label(face_label)
                if db_person:
                    print(f"[RECOG] Label {face_label} -> {db_person[1]} (DB ID: {db_person[0]})")
                else:
                    print(f"[ERROR] Label {face_label} found by face recognizer but NOT in database!")
            
            # Update person_name from database if found
            if db_person:
                person_name = db_person[1]  # name column
                person_id = db_person[0]    # id column
            else:
                person_name = "Unknown"
                person_id = None
                print(f"[DEBUG] No DB match - displaying as Unknown")
            
            # Predict mask
            face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(img_to_array(face))
            face = np.expand_dims(face, axis=0)
            
            (mask, withoutMask) = maskNet.predict(face, verbose=0)[0]
            mask_confidence = max(mask, withoutMask) * 100
            
            if mask > withoutMask:
                mask_status = "DETECTED"
                label = f"Mask: {mask_confidence:.0f}%"
                color = (0, 255, 0)
            else:
                mask_status = "NOT_DETECTED"
                label = f"No Mask: {mask_confidence:.0f}%"
                color = (0, 0, 255)
            
            # Check if person is in entry zone
            face_center_y = (startY + endY) // 2
            in_entry_zone = abs(face_center_y - zone_y) < 80
            
            # Log entry if person is recognized and in zone
            if person_name != "Unknown" and in_entry_zone and person_id is not None:
                if can_log_entry(person_id):
                    # Log to database
                    snapshot_file = save_entry_snapshot(
                        frame.copy(), person_name, person_id, mask_status, mask_confidence
                    )
                    
                    db.log_entry(
                        person_id=person_id,
                        mask_status=mask_status,
                        confidence=mask_confidence,
                        snapshot_path=snapshot_file
                    )
                    
                    entry_cooldown[person_id] = datetime.now()
                    entries_this_frame.append({
                        'name': person_name,
                        'mask_status': mask_status
                    })
                    
                    print(f"[ENTRY] {person_name} - {mask_status} - {mask_confidence:.1f}%")
            
            # Draw detection box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            
            # Display name and mask status
            display_name = person_name if person_name != "Unknown" else "Unregistered"
            
            # Debug: Show what we're displaying
            if person_name != "Unknown":
                print(f"[DISPLAY] Showing: {display_name} (from person_name: {person_name})")
            
            # Name background
            name_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (startX, startY-35), (startX+name_size[0]+10, startY), (50, 50, 50), -1)
            cv2.putText(frame, display_name, (startX+5, startY-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mask status
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (startX, endY), (startX+label_size[0]+10, endY+30), color, -1)
            cv2.putText(frame, label, (startX+5, endY+22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Entry status
            if in_entry_zone and person_name != "Unknown":
                status_text = "LOGGED" if person_id in entry_cooldown else "READY"
                cv2.putText(frame, status_text, (startX, startY-45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display stats overlay
    stats = db.get_statistics()
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (w-5, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    cv2.putText(frame, f"ORGANIZATION GATE ENTRY SYSTEM", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Registered: {stats['total_persons']} | Today: {stats['today_entries']} | " +
                      f"Compliant: {stats['mask_compliant']} | Violations: {stats['non_compliant']}", 
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Compliance Rate: {stats['compliance_rate']:.1f}% | " +
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
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
    return render_template('org_dashboard.html')

@app.route('/register')
def register_page():
    return render_template('register_person.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detection_active
    detection_active = True
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
    stats = db.get_statistics()
    today_entries = db.get_today_entries()
    
    recent_entries = []
    for entry in today_entries[:10]:
        # Handle confidence which might be stored as bytes
        confidence_val = entry[6]
        if isinstance(confidence_val, bytes):
            confidence_val = float(confidence_val.decode('utf-8'))
        elif isinstance(confidence_val, str):
            confidence_val = float(confidence_val)
        else:
            confidence_val = float(confidence_val)
            
        recent_entries.append({
            'name': entry[1],
            'employee_id': entry[2],
            'department': entry[3],
            'time': entry[4],
            'mask_status': entry[5],
            'confidence': f"{confidence_val:.1f}%",
            'snapshot': entry[7]
        })
    
    stats['recent_entries'] = recent_entries
    return jsonify(stats)

@app.route('/register_person', methods=['POST'])
def register_person():
    try:
        data = request.json
        name = data['name']
        employee_id = data['employee_id']
        department = data.get('department', '')
        role = data.get('role', '')
        phone = data.get('phone', '')
        email = data.get('email', '')
        
        # Decode base64 images
        images = []
        for i, img_data in enumerate(data['photos']):
            img_data = img_data.split(',')[1]  # Remove data:image/jpeg;base64,
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(img)
            
            # Save registration photo
            photo_filename = f"{employee_id}_{i+1}.jpg"
            photo_path = os.path.join(REGISTRATION_PHOTOS_DIR, photo_filename)
            cv2.imwrite(photo_path, img)
        
        # Register face
        success, message = face_recognizer.register_person(images, name, employee_id)
        
        if not success:
            return jsonify({'success': False, 'message': message})
        
        # Get the label (person_id) that was just assigned
        person_label = face_recognizer.label_counter - 1
        
        # Register in database
        photo_path = os.path.join(REGISTRATION_PHOTOS_DIR, f"{employee_id}_1.jpg")
        db_person_id = db.register_person(
            name=name,
            employee_id=employee_id,
            department=department,
            role=role,
            phone=phone,
            email=email,
            photo_path=photo_path,
            face_encoding=[person_label]  # Store face recognizer label
        )
        
        print(f"[DB] Registration result: person_id={db_person_id}")
        
        if db_person_id:
            return jsonify({
                'success': True,
                'message': f'Successfully registered {name}!',
                'person_id': db_person_id
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Employee ID already exists'
            })
            
    except Exception as e:
        print(f"[ERROR] Registration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_all_persons')
def get_all_persons():
    persons = db.get_all_persons()
    persons_list = []
    for p in persons:
        persons_list.append({
            'id': p[0],
            'name': p[1],
            'employee_id': p[2],
            'department': p[3],
            'role': p[4],
            'phone': p[5],
            'email': p[6],
            'registration_date': p[9]
        })
    return jsonify(persons_list)

@app.route('/get_compliance_report')
def get_compliance_report():
    report = db.get_compliance_report()
    report_list = []
    for r in report:
        report_list.append({
            'name': r[0],
            'employee_id': r[1],
            'department': r[2],
            'total_entries': r[3],
            'compliant': r[4],
            'violations': r[5],
            'compliance_rate': f"{r[6]:.1f}%"
        })
    return jsonify(report_list)

@app.route('/snapshots/<path:filename>')
def get_snapshot(filename):
    return send_from_directory(ENTRY_SNAPSHOTS_DIR, filename)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏢  ORGANIZATION GATE ENTRY SYSTEM WITH FACE RECOGNITION")
    print("="*80)
    print("📱 Dashboard: http://localhost:5000")
    print("👤 Register Person: http://localhost:5000/register")
    print("📊 Database: organization.db")
    print("📸 Entry Snapshots:", ENTRY_SNAPSHOTS_DIR)
    print("="*80 + "\n")
    app.run(debug=True, threaded=True, port=5000)
