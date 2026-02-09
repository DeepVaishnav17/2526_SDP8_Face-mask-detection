import os

# Define the path to your dataset
BASE_PATH = "dataset"

# Define where we will save the trained model
MODEL_PATH = "mask_detector.h5"

# Training settings
# LEARNING_RATE: How fast the AI learns (lower is usually better/safer)
INIT_LR = 1e-4
# EPOCHS: How many times the AI looks at the entire dataset
EPOCHS = 20
# BATCH_SIZE: How many images to look at, at one time
BS = 32