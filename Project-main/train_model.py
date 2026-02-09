import os
import cv2
import xml.etree.ElementTree as ET

# --- FIXED PATHS ---
# Make sure these match your actual folder locations
IMAGE_PATH = r"C:\Coding\SDP8_project\dataset\archive\images"
ANNOTATION_PATH = r"C:\Coding\SDP8_project\dataset\archive\annotations"
OUTPUT_PATH = r"processed_dataset" 

# Create output folders
for label in ["with_mask", "without_mask"]:
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

def extract_crops():
    count = 0
    # List the XML files in the annotation directory
    for xml_file in os.listdir(ANNOTATION_PATH):
        if not xml_file.endswith(".xml"): continue
        
        try:
            tree = ET.parse(os.path.join(ANNOTATION_PATH, xml_file))
            root = tree.getroot()
            
            # Load the corresponding image
            img_name = root.find("filename").text
            img_full_path = os.path.join(IMAGE_PATH, img_name)
            
            if not os.path.exists(img_full_path):
                # Sometimes XML has 'filename' but the file extension is different (e.g. .png vs .jpg)
                continue

            img = cv2.imread(img_full_path)
            if img is None: continue
            
            for obj in root.findall("object"):
                label = obj.find("name").text
                if label not in ["with_mask", "without_mask"]: continue
                
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                
                # Crop and Resize (Crucial for distance issues)
                face_crop = img[ymin:ymax, xmin:xmax]
                if face_crop.size == 0: continue
                
                # Resizing to 224x224 makes the AI see a "close up" 
                # even if the person was far away in the original photo
                face_crop = cv2.resize(face_crop, (224, 224))
                
                save_path = os.path.join(OUTPUT_PATH, label, f"{count}.jpg")
                cv2.imwrite(save_path, face_crop)
                count += 1
        except Exception as e:
            print(f"Skipping {xml_file} due to error: {e}")

    print(f"Extraction complete! Saved {count} face crops in {OUTPUT_PATH}")

extract_crops()