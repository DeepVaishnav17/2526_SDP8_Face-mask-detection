"""
SIMPLE WORKING MASK DETECTION SYSTEM
No complex face recognition - just detects mask/no mask with person names from database
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
import os
from datetime import datetime
from database import OrganizationDatabase

app = Flask(__name__)

# Initialize
db = OrganizationDatabase()
print("[INFO] Loading models...")
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector_best.h5")
print("[INFO] Models loaded!")

camera = None
detection_active = False

# Load reference faces from database
reference_faces = {}

def load_reference_faces():
    """Load all registered persons' faces from database"""
    global reference_faces
    reference_faces = {}
    
    persons = db.get_all_persons()
    for person in persons:
        person_id = person[0]
        name = person[1]
        photo_path = person[7] if len(person) > 7 else None
        
        if photo_path and os.path.exists(photo_path):
            img = cv2.imread(photo_path)
            if img is not None:
                # Resize to standard size
                img = cv2.resize(img, (100, 100))
                # Convert to grayscale for histogram
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Calculate histogram
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist)
                reference_faces[name] = hist
                print(f"[INFO] Loaded reference face for {name}")

# Load faces on startup
load_reference_faces()

def recognize_person(face_img):
    """Recognize person using histogram comparison"""
    global reference_faces
    
    if len(reference_faces) == 0:
        return "Unknown", 0
    
    try:
        # Resize to standard size
        face_resized = cv2.resize(face_img, (100, 100))
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        
        # Compare with all reference faces
        best_match = "Unknown"
        best_score = float('inf')
        
        for name, ref_hist in reference_faces.items():
            score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CHISQR)
            if score < best_score:
                best_score = score
                best_match = name
        
        # Threshold for recognition (lower is better, 500 is lenient)
        if best_score < 500:
            confidence = max(0, min(100, int(100 - (best_score / 10))))
            return best_match, confidence
        
        return "Unknown", 0
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
        return "Unknown", 0


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            if endX - startX < 50 or endY - startY < 50:
                continue
                
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(img_to_array(face))
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, verbose=0)
    
    return (locs, preds)

def generate_frames():
    global camera
    
    while detection_active:
        if camera is None:
            camera = cv2.VideoCapture(0)
            
        success, frame = camera.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        # Get all registered persons
        persons = db.get_all_persons()
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            # Extract face for recognition
            face_img = frame[startY:endY, startX:endX]
            
            # Recognize person
            person_name = "Unknown"
            recog_conf = 0
            if face_img.size > 0:
                person_name, recog_conf = recognize_person(face_img)
            
            # Determine mask status
            if mask > withoutMask:
                mask_label = "MASK"
                color = (0, 255, 0)  # Green
            else:
                mask_label = "NO MASK"
                color = (0, 0, 255)  # Red
            
            mask_confidence = max(mask, withoutMask) * 100
            
            # Draw box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            
            # Draw person name (top)
            name_text = person_name if person_name != "Unknown" else "Unregistered"
            name_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (startX, startY-60), (startX+name_size[0]+10, startY-30), (50,50,50), -1)
            cv2.putText(frame, name_text, (startX+5, startY-38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Draw mask status (below name)
            mask_text = f"{mask_label}: {mask_confidence:.0f}%"
            cv2.rectangle(frame, (startX, startY-30), (startX+200, startY), color, -1)
            cv2.putText(frame, mask_text, (startX+5, startY-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Show stats in corner
        cv2.rectangle(frame, (10, 10), (300, 50), (0,0,0), -1)
        cv2.putText(frame, f"Registered: {len(reference_faces)}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('simple_dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({'status': 'started'})

@app.route('/stop_detection')
def stop_detection():
    global detection_active, camera
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

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
            'photo': p[7] if len(p) > 7 else ''
        })
    return jsonify(persons_list)

@app.route('/register')
def register_page():
    return render_template('register_person.html')

@app.route('/register_person', methods=['POST'])
def register_person():
    try:
        data = request.json
        name = data['name']
        employee_id = data['employee_id']
        department = data.get('department', '')
        role = data.get('role', '')
        
        # Save first photo
        img_data = data['photos'][0].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        photo_filename = f"{employee_id}_1.jpg"
        photo_path = os.path.join('registration_photos', photo_filename)
        os.makedirs('registration_photos', exist_ok=True)
        cv2.imwrite(photo_path, img)
        
        # Register in database
        db_person_id = db.register_person(
            name=name,
            employee_id=employee_id,
            department=department,
            role=role,
            phone='',
            email='',
            photo_path=photo_path,
            face_encoding=[]
        )
        
        if db_person_id:
            # Reload reference faces
            load_reference_faces()
            return jsonify({'success': True, 'message': f'Registered {name}!'})
        else:
            return jsonify({'success': False, 'message': 'Employee ID exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎭  SIMPLE MASK DETECTION SYSTEM")
    print("="*80)
    print("📱 Dashboard: http://localhost:5000")
    print("👤 Register: http://localhost:5000/register")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
