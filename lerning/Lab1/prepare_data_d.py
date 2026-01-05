import os
import cv2
import xml.etree.ElementTree as ET

# --- PATHS ---
base_dir = os.getcwd() # Get current working directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

images_path = os.path.join(  ROOT_DIR, "dataset", "archive", "images")
xmls_path = os.path.join(ROOT_DIR, "dataset", "archive", "annotations")

output_mask_path = os.path.join(base_dir, "dataset", "with_mask")
output_no_mask_path = os.path.join(base_dir, "dataset", "without_mask")

# Create output directories
os.makedirs(output_mask_path, exist_ok=True)
os.makedirs(output_no_mask_path, exist_ok=True)

print("[INFO] Starting data processing...")
print(f"[DEBUG] Scanning for images in: {images_path}")

count_mask = 0
count_nomask = 0

# Check if folders exist
if not os.path.exists(images_path) or not os.path.exists(xmls_path):
    print("[CRITICAL ERROR] The 'archive' folder paths are wrong.")
    print(f"I expected to find: {images_path}")
    print("Check your folder structure in VS Code!")
    exit()

files = os.listdir(xmls_path)
print(f"[DEBUG] Found {len(files)} XML files. Processing...")

for xml_file in files:
    if not xml_file.endswith(".xml"):
        continue

    # Parse XML
    try:
        tree = ET.parse(os.path.join(xmls_path, xml_file))
        root = tree.getroot()
        filename = root.find('filename').text
    except Exception as e:
        print(f"[SKIP] Corrupt XML {xml_file}: {e}")
        continue

    # --- ROBUST PATH FIXING ---
    # 1. Try exact filename
    img_full_path = os.path.join(images_path, filename)
    
    # 2. If not found, try adding extensions (common dataset bug)
    if not os.path.exists(img_full_path):
        if os.path.exists(img_full_path + ".png"):
            img_full_path += ".png"
        elif os.path.exists(img_full_path + ".jpg"):
            img_full_path += ".jpg"
        elif os.path.exists(img_full_path + ".jpeg"):
            img_full_path += ".jpeg"
            
    # 3. If still not found, try swapping extensions
    if not os.path.exists(img_full_path):
        if "png" in filename: 
            img_full_path = img_full_path.replace("png", "jpg")
        elif "jpg" in filename: 
            img_full_path = img_full_path.replace("jpg", "png")

    # Load Image
    image = cv2.imread(img_full_path)
    
    if image is None:
        # Silently skip missing images to avoid cluttering terminal
        continue

    (h, w) = image.shape[:2]

    # Process faces
    for member in root.findall('object'):
        label = member.find('name').text
        bndbox = member.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Fix coordinates (sometimes they are negative in bad data)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        # Crop
        face = image[ymin:ymax, xmin:xmax]

        # Skip empty crops
        if face.size == 0: continue

        # Save
        if label == 'with_mask':
            p = os.path.join(output_mask_path, f"mask_{count_mask}.jpg")
            cv2.imwrite(p, face)
            count_mask += 1
        elif label == 'without_mask':
            p = os.path.join(output_no_mask_path, f"nomask_{count_nomask}.jpg")
            cv2.imwrite(p, face)
            count_nomask += 1

print("------------------------------------------------")
print(f"[SUCCESS] Processing Finished.")
print(f"Mask Images: {count_mask}")
print(f"No Mask Images: {count_nomask}")
print("------------------------------------------------")
print("NOW YOU CAN RUN 'python train_model.py'")