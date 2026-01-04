import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt # NEW: For efficiency graphs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- 1. CONFIGURATION ---
IMAGE_PATH = r"C:\Coding\SDP8_project\dataset\archive\images"
ANNOTATION_PATH = r"C:\Coding\SDP8_project\dataset\archive\annotations"
OUTPUT_PATH = r"processed_dataset"

for label in ["with_mask", "without_mask"]:
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

# --- 2. EXTRACT FACE CROPS (Same logic, ensured resizing) ---
print("[INFO] Extracting face crops...")
count = 0
for xml_file in os.listdir(ANNOTATION_PATH):
    if not xml_file.endswith(".xml"): continue
    try:
        tree = ET.parse(os.path.join(ANNOTATION_PATH, xml_file))
        root = tree.getroot()
        img_name = root.find("filename").text
        img_full_path = os.path.join(IMAGE_PATH, img_name)
        if not os.path.exists(img_full_path): continue
        img = cv2.imread(img_full_path)
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in ["with_mask", "without_mask"]: continue
            bbox = obj.find("bndbox")
            xmin, ymin = int(bbox.find("xmin").text), int(bbox.find("ymin").text)
            xmax, ymax = int(bbox.find("xmax").text), int(bbox.find("ymax").text)
            face_crop = img[ymin:ymax, xmin:xmax]
            if face_crop.size == 0: continue
            face_crop = cv2.resize(face_crop, (224, 224)) # Normalized for all distances
            cv2.imwrite(os.path.join(OUTPUT_PATH, label, f"{count}.jpg"), face_crop)
            count += 1
    except Exception as e: print(f"Error: {e}")

# --- 3. ADVANCED AUGMENTATION (Fixes Distance/Hand Issues) ---
# We add zoom to simulate distance and brightness to handle shadows
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.3,          # FIX: Helps with distance
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3], # FIX: Helps with visibility
    fill_mode="nearest",
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(OUTPUT_PATH, target_size=(224, 224), 
                                        batch_size=32, subset="training")
val_gen = datagen.flow_from_directory(OUTPUT_PATH, target_size=(224, 224), 
                                      batch_size=32, subset="validation")

# --- 4. MODEL BUILDING ---
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers: layer.trainable = False

# --- 5. TRAINING & EFFICIENCY PLOTTING ---
INIT_LR = 1e-4
EPOCHS = 15 # Increased for better learning
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model...")
H = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save the model
model.save("mask_detector.h5")

# --- 6. PLOT EFFICIENCY GRAPH (Include this in your report!) ---
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
print("[INFO] Efficiency plot saved as plot.png")