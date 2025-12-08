# Category 1: Two-Stage Detectors (The "Beginner Friendly" Way)
This is the method I recommended in the plan. It is not "direct" in one single math step, but it is easier to build and debug.

Step 1: Find the Face.

Step 2: Check that specific face for a Mask.

Algorithms:
- Haar Cascades
    - Role in Project: Finds the face.
    - Why use it? : It is incredibly fast and pre-installed in OpenCV. It doesn't need a GPU.

- MobileNetV2
    - Role in Project: Checks for mask.
    - Why use it? : It is a "lightweight" Deep Learning model designed specifically for mobile phones and laptops. It is fast and accurate enough (99%) for this task.

- VGG16 / ResNet50
    - Role in Project: Checks for mask.
    - Why use it? : These are "heavy" models. They are slightly more accurate than MobileNet but run much slower. Avoid these if you want real-time video on a laptop.

# Category 2: One-Stage Detectors (The "Direct" Way)
These algorithms look at the image once and immediately draw boxes around "Masked Faces" and "Unmasked Faces" at the same time. They are more advanced and "cooler," but harder to set up for beginners.

1. YOLO (You Only Look Once)
What it is: The most famous object detection algorithm in the world.

Versions: YOLOv5 or YOLOv8 (specifically the "Nano" or "Small" versions) are best for laptops.

Pros: Extremely fast. It detects the face and the mask in a single millisecond.

Cons: It requires a different data format (YOLO format: .txt files with coordinates) which can be annoying to prepare. If you get an installation error, it is harder to fix than standard TensorFlow.

2. SSD (Single Shot MultiBox Detector)
What it is: A competitor to YOLO. Usually paired with MobileNet (called SSD-MobileNet).

Pros: faster than YOLO on older hardware (like Raspberry Pi or old laptops).

Cons: Slightly less accurate than YOLO.


# Visual Comparison
Option A (Two-Stage - Recommended)

[Camera Input] -> [Haar Cascade] finds face at (x,y) -> [Cut out Face] -> [MobileNetV2] says "99% Mask" -> [Draw Green Box]

Option B (One-Stage - YOLO)

[Camera Input] -> [YOLOv8] -> "I see a 'Masked-Face' at (x,y)" -> [Draw Green Box]