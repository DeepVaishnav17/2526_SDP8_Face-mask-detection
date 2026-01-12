"""
IMPROVED MASK DETECTION MODEL TRAINING
Incorporates best practices for higher accuracy
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = r"C:\Users\mrudu\OneDrive\Desktop\ML_Project\dataset\Train"
MODEL_SAVE_PATH = "mask_detector_improved.h5"
BEST_MODEL_PATH = "mask_detector_best.h5"

# Training Parameters
INIT_LR = 1e-4
EPOCHS = 30  # Increased from 15
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Model Selection: 'mobilenet' or 'efficientnet'
BASE_MODEL_TYPE = 'mobilenet'  # Change to 'efficientnet' for better accuracy

print("="*60)
print("🚀 IMPROVED MASK DETECTION MODEL TRAINING")
print("="*60)

# ============================================================
# 1. VERIFY DATASET
# ============================================================
print("\n[INFO] Checking dataset...")
try:
    with_mask_count = len(os.listdir(os.path.join(DATASET_PATH, 'WithMask')))
    without_mask_count = len(os.listdir(os.path.join(DATASET_PATH, 'WithoutMask')))
    total_images = with_mask_count + without_mask_count
    
    print(f"✅ WithMask: {with_mask_count} images")
    print(f"✅ WithoutMask: {without_mask_count} images")
    print(f"✅ Total: {total_images} images")
    print(f"✅ Balance Ratio: {with_mask_count/without_mask_count:.2f}:1")
    
    if total_images < 1000:
        print("⚠️  WARNING: Dataset is small. Consider collecting more images for better accuracy.")
except Exception as e:
    print(f"❌ ERROR: Cannot access dataset at {DATASET_PATH}")
    print(f"Error: {e}")
    exit(1)

# ============================================================
# 2. ENHANCED DATA AUGMENTATION
# ============================================================
print("\n[INFO] Setting up enhanced data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,              # Increased from 20
    zoom_range=0.4,                 # Increased from 0.3
    width_shift_range=0.25,         # Increased from 0.2
    height_shift_range=0.25,
    shear_range=0.2,                # Increased from 0.15
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],    # Wider range
    channel_shift_range=30,         # NEW: Color variation
    fill_mode="nearest",
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data
train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Load validation data
val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # Important for evaluation
)

print(f"✅ Training samples: {train_gen.samples}")
print(f"✅ Validation samples: {val_gen.samples}")
print(f"✅ Classes: {list(train_gen.class_indices.keys())}")

# ============================================================
# 3. CALCULATE CLASS WEIGHTS (Handle Imbalance)
# ============================================================
print("\n[INFO] Calculating class weights...")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Class weights: {class_weight_dict}")

# ============================================================
# 4. BUILD IMPROVED MODEL
# ============================================================
print(f"\n[INFO] Building model with {BASE_MODEL_TYPE.upper()} base...")

# Select base model
if BASE_MODEL_TYPE == 'efficientnet':
    baseModel = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )
    print("✅ Using EfficientNetB0 (Better accuracy)")
else:
    baseModel = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )
    print("✅ Using MobileNetV2 (Faster)")

# Build improved head model with more layers
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)

# Multiple dense layers for better feature learning
headModel = Dense(256, activation="relu", name="dense_1")(headModel)
headModel = BatchNormalization(name="bn_1")(headModel)
headModel = Dropout(0.5, name="dropout_1")(headModel)

headModel = Dense(128, activation="relu", name="dense_2")(headModel)
headModel = BatchNormalization(name="bn_2")(headModel)
headModel = Dropout(0.4, name="dropout_2")(headModel)

headModel = Dense(64, activation="relu", name="dense_3")(headModel)
headModel = Dropout(0.3, name="dropout_3")(headModel)

headModel = Dense(2, activation="softmax", name="output")(headModel)

# Create final model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model initially
for layer in baseModel.layers:
    layer.trainable = False

print(f"✅ Model created with {len(model.layers)} layers")
print(f"✅ Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

# ============================================================
# 5. SETUP CALLBACKS
# ============================================================
print("\n[INFO] Setting up training callbacks...")

# Early stopping - prevents overfitting
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction - improves convergence
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Model checkpoint - saves best model
checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]
print("✅ Callbacks configured: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")

# ============================================================
# 6. PHASE 1: INITIAL TRAINING (Frozen Base)
# ============================================================
print("\n" + "="*60)
print("PHASE 1: INITIAL TRAINING (Base Model Frozen)")
print("="*60)

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=INIT_LR),
    metrics=["accuracy"]
)

print(f"[INFO] Training for {EPOCHS} epochs (will stop early if no improvement)...")
H1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n✅ Phase 1 completed!")
print(f"Best validation accuracy: {max(H1.history['val_accuracy']):.4f}")

# ============================================================
# 7. PHASE 2: FINE-TUNING (Unfreeze Top Layers)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: FINE-TUNING (Unfreezing Top Layers)")
print("="*60)

# Unfreeze last 20 layers for fine-tuning
for layer in baseModel.layers[-20:]:
    layer.trainable = True

print(f"✅ Unfrozen last 20 layers of base model")
print(f"✅ Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

# Recompile with lower learning rate
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-5),  # 10x lower LR
    metrics=["accuracy"]
)

print(f"[INFO] Fine-tuning for 10 more epochs...")
H2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n✅ Phase 2 completed!")
print(f"Best validation accuracy: {max(H2.history['val_accuracy']):.4f}")

# ============================================================
# 8. SAVE FINAL MODEL
# ============================================================
print("\n[INFO] Saving final model...")
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
print(f"✅ Best model saved to {BEST_MODEL_PATH}")

# ============================================================
# 9. EVALUATION & METRICS
# ============================================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Load best model for evaluation
from tensorflow.keras.models import load_model
best_model = load_model(BEST_MODEL_PATH)

# Get predictions
print("[INFO] Generating predictions on validation set...")
predictions = best_model.predict(val_gen, verbose=1)
pred_classes = np.argmax(predictions, axis=1)
true_classes = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# Classification Report
print("\n📊 CLASSIFICATION REPORT:")
print("="*60)
report = classification_report(
    true_classes,
    pred_classes,
    target_names=class_labels,
    digits=4
)
print(report)

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write("MASK DETECTION MODEL - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(report)
print("✅ Classification report saved to classification_report.txt")

# Confusion Matrix
print("\n[INFO] Generating confusion matrix...")
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Mask Detection Model', fontsize=16, pad=20)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("✅ Confusion matrix saved to confusion_matrix.png")

# ============================================================
# 10. PLOT TRAINING HISTORY
# ============================================================
print("\n[INFO] Plotting training history...")

# Combine both training phases
total_epochs_phase1 = len(H1.history['loss'])
total_epochs_phase2 = len(H2.history['loss'])

all_train_loss = H1.history['loss'] + H2.history['loss']
all_val_loss = H1.history['val_loss'] + H2.history['val_loss']
all_train_acc = H1.history['accuracy'] + H2.history['accuracy']
all_val_acc = H1.history['val_accuracy'] + H2.history['val_accuracy']

epochs_range = range(len(all_train_loss))

# Create subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Loss
ax1.plot(epochs_range, all_train_loss, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs_range, all_val_loss, 'r-', label='Validation Loss', linewidth=2)
ax1.axvline(x=total_epochs_phase1, color='green', linestyle='--', 
            label='Fine-tuning Starts', linewidth=1.5)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot Accuracy
ax2.plot(epochs_range, all_train_acc, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(epochs_range, all_val_acc, 'r-', label='Validation Accuracy', linewidth=2)
ax2.axvline(x=total_epochs_phase1, color='green', linestyle='--',
            label='Fine-tuning Starts', linewidth=1.5)
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
print("✅ Training history saved to training_history.png")

# ============================================================
# 11. FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 FINAL RESULTS:")
print(f"   • Best Validation Accuracy: {max(all_val_acc):.4f} ({max(all_val_acc)*100:.2f}%)")
print(f"   • Final Training Accuracy: {all_train_acc[-1]:.4f} ({all_train_acc[-1]*100:.2f}%)")
print(f"   • Total Epochs Trained: {len(all_train_loss)}")
print(f"   • Phase 1 Epochs: {total_epochs_phase1}")
print(f"   • Phase 2 Epochs: {total_epochs_phase2}")

print(f"\n💾 SAVED FILES:")
print(f"   • Final Model: {MODEL_SAVE_PATH}")
print(f"   • Best Model: {BEST_MODEL_PATH}")
print(f"   • Training History: training_history.png")
print(f"   • Confusion Matrix: confusion_matrix.png")
print(f"   • Classification Report: classification_report.txt")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review the confusion matrix and classification report")
print(f"   2. Test the model on real webcam feed")
print(f"   3. If accuracy is low, consider:")
print(f"      - Collecting more diverse training data")
print(f"      - Trying BASE_MODEL_TYPE = 'efficientnet'")
print(f"      - Adjusting CONFIDENCE_MIN threshold in web_app.py")

print("\n" + "="*60)
