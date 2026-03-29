# Lab 1

## Date: December 8, 2025

### Today's Task

Exploration, understanding, and documentation of the chosen Face Mask Detection Algorithm and its underlying mathematical principles for production deployment.

# By Mrudul

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

# By Deep

# Algorithm Deep Dive & Technical Learning Log

---

### 1. Algorithm Selection: MobileNetV2 (CNN)

The selected algorithm for the classification task is **MobileNetV2**, an advanced, lightweight **Convolutional Neural Network (CNN)**.

#### 1.1 Key Principles Explored:

- **Feature Extraction via Convolution:** Explored how the basic **convolution operation** uses small filters (kernels) to slide over images and automatically learn a hierarchy of features, starting from simple edges to complex shapes (like the face and mask).
- **Efficiency:** Understood the core innovation of MobileNetV2: **Depthwise Separable Convolution**. This technique replaces a computationally heavy standard convolution with two much smaller steps (Depthwise and Pointwise convolution), drastically reducing the model's size and computational demand, making it ideal for real-time edge deployment.
- **Learning Strategy:** Confirmed the use of **Transfer Learning**, where the MobileNetV2 Base Model (pre-trained on ImageNet) provides generalized visual knowledge, allowing us to only train a small, specific classification head on our mask data.

---

### 2. Core Mathematical & Technical Components

Explored the functions and mathematics governing the training loop within `train_model.py`:

| Component         | Technical Role & Learning | Why it Matters                                                                                                                                                                                                                       |
| :---------------- | :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Loss Function** | **Binary Cross-Entropy**  | Measures the model's error by comparing the predicted probabilities to the true labels (One-Hot Encoded: [1, 0] or [0, 1]). The training goal is solely to **minimize this value**.                                                  |
| **Optimizer**     | **Adam**                  | Learned that Adam is a sophisticated version of Gradient Descent. It uses the calculated loss to determine the optimal direction and size of the step (governed by the **Learning Rate**) to update the model's millions of weights. |
| **Output Layer**  | **Softmax Activation**    | Understood that the final dense layer uses Softmax to convert the model's raw scores into a final set of **mutually exclusive probabilities** that sum to 1 (e.g., 90% chance of mask, 10% chance of no mask).                       |
| **Data Encoding** | **One-Hot Encoding**      | Learned that categorical labels ("with_mask") must be mathematically converted into vectors (e.g., [1, 0]) for the Binary Cross-Entropy loss function to work correctly.                                                             |

---

### 3. Conclusion

The chosen architecture and training methodology provide a strong balance between high performance (due to Transfer Learning) and low latency (due to MobileNetV2's efficient design), aligning with the requirements for a production-grade detection system.


