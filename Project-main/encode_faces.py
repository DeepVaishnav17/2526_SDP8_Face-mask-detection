import face_recognition
import argparse
import pickle
import cv2
import os

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", default="encodings.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default="hog",
	help="face detection model to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = []
for root, dirs, files in os.walk(args["dataset"]):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            imagePaths.append(os.path.join(root, file))

knownEncodings = []
knownNames = []

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# Dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(args["encodings"], "wb") as f:
	f.write(pickle.dumps(data))
print(f"[INFO] encodings saved to {args['encodings']}")
