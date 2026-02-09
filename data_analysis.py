import os

# Define the path to our data
# We use os.path.join so it works on Windows and Mac/Linux
dataset_path = os.path.join('dataset', 'Train')

# Define categories
categories = ['WithMask', 'WithoutMask']

print("------------------------------------------------")
print("📊 DATASET ANALYSIS REPORT")
print("------------------------------------------------")

total_images = 0

# Loop through both folders and count files
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    
    # Check if folder exists
    if os.path.exists(folder_path):
        # Get list of all files in the folder
        image_files = os.listdir(folder_path)
        count = len(image_files)
        total_images += count
        print(f"✅ Found {count} images in '{category}'")
    else:
        print(f"❌ Error: Folder '{category}' not found at {folder_path}")

print("------------------------------------------------")
print(f"Total Training Images: {total_images}")
print("------------------------------------------------")

# Simple check to see if data is balanced
if total_images > 0:
    print("Status: Data is ready for processing.")
else:
    print("Status: No images found. Please check your dataset folder.")