import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your trained model and the dataset
model = load_model("mask_detector.h5")
DIRECTORY = "processed_dataset" # The folder with your face crops

# 2. Prepare the Validation Data Generator
# We use the SAME settings as your training script to ensure consistency
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False # Crucial for matching predictions to labels
)

# 3. Measure Validation Accuracy
print("[INFO] Evaluating model on validation data...")
predictions = model.predict(val_gen)
pred_idxs = np.argmax(predictions, axis=1) # Get the index of the highest probability

# 4. Generate the Classification Report (Precision, Recall, F1-Score)
print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(val_gen.classes, pred_idxs, 
                               target_names=val_gen.class_indices.keys())
print(report)

# 5. Generate and Plot Confusion Matrix
cm = confusion_matrix(val_gen.classes, pred_idxs)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=val_gen.class_indices.keys(), 
            yticklabels=val_gen.class_indices.keys())
plt.title("Efficiency Matrix: Predicted vs Actual")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.savefig("efficiency_matrix.png")
print("[INFO] Confusion Matrix saved as efficiency_matrix.png")