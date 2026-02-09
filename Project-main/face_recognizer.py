"""
Simple Face Recognition using OpenCV with ORB features
Works with standard opencv-python (no opencv-contrib needed)
"""

import cv2
import numpy as np
import pickle
import os
from datetime import datetime

class SimpleFaceRecognizer:
    """Face recognition using ORB feature matching"""
    
    def __init__(self, model_path="face_database.pkl"):
        self.model_path = model_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.orb = cv2.ORB_create(nfeatures=500)
        self.person_database = {}  # {label: {'name': str, 'employee_id': str, 'features': [descriptors]}}
        self.label_counter = 0
        
        # Load existing database if available
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                    self.person_database = data['database']
                    self.label_counter = data['counter']
                print(f"[FR] Loaded {len(self.person_database)} registered persons")
            except Exception as e:
                print(f"[FR] Could not load database: {e}, starting fresh")
    
    def extract_face_and_features(self, image):
        """Extract face and compute ORB features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))  # Standardize size
        
        # Extract ORB features
        keypoints, descriptors = self.orb.detectAndCompute(face, None)
        
        return face, descriptors
    
    def register_person(self, images, name, employee_id):
        """
        Register a new person with multiple images
        """
        all_descriptors = []
        
        # Extract features from all images
        for img in images:
            _, descriptors = self.extract_face_and_features(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        if len(all_descriptors) == 0:
            return False, "No faces detected in images"
        
        # Store person info with features
        self.person_database[self.label_counter] = {
            'name': name,
            'employee_id': employee_id,
            'features': all_descriptors,
            'registered_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"[FR] Registering {name} with {len(all_descriptors)} feature sets")
        
        # Save database
        with open(self.model_path, "wb") as f:
            pickle.dump({
                'database': self.person_database,
                'counter': self.label_counter + 1
            }, f)
        
        self.label_counter += 1
        return True, f"Successfully registered {name}"
    
    def recognize_face(self, image, confidence_threshold=5):
        """
        Recognize person from image using feature matching
        Returns: (person_id, name, confidence) or (None, "Unknown", 0)
        """
        _, descriptors = self.extract_face_and_features(image)
        
        if descriptors is None:
            print(f"[FR_DEBUG] No face/descriptors extracted from image")
            return None, "Unknown", 0
        
        print(f"[FR_DEBUG] Extracted {len(descriptors)} descriptors from current face")
        
        if len(self.person_database) == 0:
            print(f"[FR_DEBUG] Person database is EMPTY!")
            return None, "Unknown", 0
        
        print(f"[FR_DEBUG] Comparing against {len(self.person_database)} registered persons")
        
        # Match against all registered persons
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_match_label = None
        best_match_score = 0
        
        for label, person_data in self.person_database.items():
            total_matches = 0
            total_comparisons = 0
            
            # Compare with all feature sets for this person
            for stored_descriptors in person_data['features']:
                if stored_descriptors is not None and len(stored_descriptors) > 0:
                    try:
                        matches = bf.match(descriptors, stored_descriptors)
                        # Count good matches (distance < 70 for more lenient matching)
                        good_matches = [m for m in matches if m.distance < 70]
                        total_matches += len(good_matches)
                        total_comparisons += 1
                    except Exception as e:
                        print(f"[FR_DEBUG] Match error for label {label}: {e}")
                        continue
            
            # Calculate average match score
            if total_comparisons > 0:
                avg_matches = total_matches / total_comparisons
                print(f"[FR_DEBUG] Label {label} ({person_data['name']}): {avg_matches:.1f} avg matches (threshold: {confidence_threshold})")
                if avg_matches > best_match_score:
                    best_match_score = avg_matches
                    best_match_label = label
        
        # Check if best match meets threshold (lowered to 15 for better recognition)
        if best_match_label is not None and best_match_score >= confidence_threshold:
            person_info = self.person_database[best_match_label]
            confidence = min(100, int((best_match_score / confidence_threshold) * 100))
            print(f"[FR_DEBUG] ✓ MATCHED! Label {best_match_label}: {person_info['name']} with {best_match_score:.1f} score")
            return best_match_label, person_info['name'], confidence
        
        print(f"[FR_DEBUG] ✗ NO MATCH - Best score: {best_match_score:.1f} < threshold: {confidence_threshold}")
        return None, "Unknown", 0
    
    def get_all_persons(self):
        """Get list of all registered persons"""
        return [(label, info['name'], info['employee_id']) 
                for label, info in self.person_database.items()]
    
    def delete_person(self, label):
        """Remove a person from the system"""
        if label in self.person_database:
            del self.person_database[label]
            
            # Save updated database
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    'database': self.person_database,
                    'counter': self.label_counter
                }, f)
            
            return True
        return False


# Test code
if __name__ == "__main__":
    recognizer = SimpleFaceRecognizer()
    print("Face Recognizer initialized!")
    print(f"Registered persons: {len(recognizer.person_database)}")
    
    for label, info in recognizer.person_database.items():
        print(f"  - {info['name']} ({info['employee_id']})")
