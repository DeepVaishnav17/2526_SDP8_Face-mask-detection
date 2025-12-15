# # train_model.py
# import tensorflow as tf

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import config_d as config_d # Import your config settings
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import config_d


print("[INFO] loading images...")

data = []
labels = []

# 1. LOAD THE DATA
# We loop through the "with_mask" and "without_mask" folders
categories = ["with_mask", "without_mask"]

for category in categories:
    path = os.path.join(config_d.BASE_PATH, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            # Load image, resize to 224x224 (MobileNet standard)
            image = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            image = tf.keras.utils.img_to_array(image)
            image = preprocess_input(image) # Clean the image for the model

            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"Skipping file {img_name}: {e}")

# Convert data to numpy arrays (what the AI understands)
data = np.array(data, dtype="float32")
labels = np.array(labels)

# 2. PREPARE LABELS
# Convert "with_mask" to [1, 0] and "without_mask" to [0, 1]
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data: 80% for training, 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 3. BUILD THE MODEL (MobileNetV2)
print("[INFO] building model...")
# Load MobileNetV2 without the top layer (we will add our own)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Add our custom layers
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Combine them
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers (so we don't ruin the pre-training)
for layer in baseModel.layers:
	layer.trainable = False

# 4. TRAIN
print("[INFO] compiling model...")
opt = Adam(learning_rate=config_d.INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Image augmentation (creates fake variations of your images to train better)
aug = ImageDataGenerator(
	rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2,
	shear_range=0.15, horizontal_flip=True,
	fill_mode="nearest")

print("[INFO] training...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=config_d.BS),
	steps_per_epoch=len(trainX) // config_d.BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // config_d.BS,
	epochs=config_d.EPOCHS)

# 5. SAVE
print(f"[INFO] saving mask detector model to {config_d.MODEL_PATH}...")
model.save(config_d.MODEL_PATH)
print("Done! You are ready for production.")