# Real-Time Face Mask Detection System

## Project Overview

This project is a professional-grade Computer Vision solution designed to automate face mask compliance monitoring. The system utilizes Deep Learning to detect individuals in a video stream and classify whether they are wearing a protective mask. It is optimized for real-time performance on standard hardware through the use of efficient neural network architectures and multi-threaded programming.

## Technical Architecture

The system operates on a dual-model pipeline to ensure high accuracy and low latency:

1.  **Face Detection (Localization):** Uses a Single Shot MultiBox Detector (SSD) with a ResNet-10 backbone to identify facial coordinates in real-time.
2.  **Mask Classification (Recognition):** Uses **MobileNetV2**, a specialized architecture employing Depthwise Separable Convolutions to reduce computational complexity by approximately 9x compared to standard convolutions.

## Key Features

- **Multi-threaded Execution:** Implements a background thread for frame acquisition to eliminate camera lag and maintain 30+ FPS.
- **Temporal Smoothing:** Employs a 10-frame sliding window average to stabilize predictions and prevent flickering of detection labels.
- **Automated Audit Logs:** Generates a `mask_violations.csv` file with timestamps and confidence scores for every detected violation.
- **Audio Deterrent:** Integrated alert system using the motherboard speaker to provide immediate feedback for non-compliance.

## Model Evaluation

The classifier was trained using transfer learning and evaluated on a validation dataset to ensure generalized performance.

| Metric                  | Result |
| :---------------------- | :----- |
| **Training Accuracy**   | 88%    |
| **Validation Accuracy** | 87%    |
| **Inference Time**      | ~50ms  |

## Installation

### Dependencies

Ensure you have Python 3.8+ installed. The following libraries are required:

- TensorFlow / Keras
- OpenCV
- NumPy

### Setup

```bash
pip install tensorflow opencv-python numpy
```
