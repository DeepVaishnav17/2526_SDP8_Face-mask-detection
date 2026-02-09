# Face Mask Detection System - Complete Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Alternative Installation Methods](#alternative-installation-methods)

---

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: 4GB (8GB recommended)
- **CPU**: Dual-core processor (Quad-core recommended)
- **Storage**: 2GB free space
- **Webcam**: Built-in or USB webcam (720p or higher)
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **Internet**: Required for initial setup and package downloads

### Recommended Requirements

- **RAM**: 8GB or more
- **CPU**: Quad-core processor or better
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **Webcam**: 1080p webcam for better accuracy
- **Python**: 3.10 or 3.11

### Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | ✅ Fully Supported | Tested on Windows 11 |
| macOS 10.14+ | ✅ Fully Supported | Both Intel and Apple Silicon |
| Ubuntu 18.04+ | ✅ Fully Supported | Also works on Debian-based distros |
| Raspberry Pi 4 | ⚠️ Limited | Requires optimization for real-time performance |

---

## Prerequisites

### 1. Python Installation

#### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation:

```bash
python --version
# Should output: Python 3.x.x
```

#### macOS

```bash
# Using Homebrew (recommended)
brew install python@3.11

# Verify
python3 --version
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

### 2. Git (Optional but Recommended)

#### Windows
Download from [git-scm.com](https://git-scm.com/download/win)

#### macOS
```bash
brew install git
```

#### Linux
```bash
sudo apt install git
```

### 3. Webcam Drivers

- **Windows**: Usually installed automatically
- **Linux**: Install v4l-utils
  ```bash
  sudo apt install v4l-utils
  # Test webcam
  v4l2-ctl --list-devices
  ```

---

## Installation Steps

### Step 1: Download the Project

#### Option A: Using Git (Recommended)

```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection/Project-main
```

#### Option B: Manual Download

1. Download the project ZIP file
2. Extract to your desired location
3. Navigate to the `Project-main` folder

### Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies from your system Python.

#### Windows

```bash
# Navigate to project directory
cd Project-main

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

#### macOS/Linux

```bash
# Navigate to project directory
cd Project-main

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

**Note**: You need to activate the virtual environment every time you open a new terminal.

### Step 3: Install Dependencies

With the virtual environment activated:

```bash
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements_web.txt
```

**Expected Installation Time**: 3-5 minutes (depending on internet speed)

### Step 4: Verify Installation

Check that all critical packages are installed:

```bash
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

**Expected Output**:
```
Flask: 3.0.0
OpenCV: 4.12.0.88
TensorFlow: 2.20.0
NumPy: 2.2.6
```

### Step 5: Verify Model Files

Ensure all model files are present in the `Project-main` folder:

```bash
# Windows PowerShell
ls mask_detector_best.h5, deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel

# macOS/Linux
ls -lh mask_detector_best.h5 deploy.prototxt res10_300x300_ssd_iter_140000.caffemodel
```

**Required Files**:
- `mask_detector_best.h5` (~26MB) - Mask classification model
- `deploy.prototxt` (~28KB) - Face detector configuration
- `res10_300x300_ssd_iter_140000.caffemodel` (~10MB) - Face detector model

**Missing Files?**
- Download from the project repository's releases page
- Or contact the project maintainer

### Step 6: Test Webcam Access

```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam:', 'OK' if cap.isOpened() else 'FAILED'); cap.release()"
```

**Expected Output**: `Webcam: OK`

If you see `FAILED`, check [Troubleshooting](#troubleshooting).

### Step 7: Run the Application

```bash
python web_app.py
```

**Expected Output**:
```
[INFO] Loading AI models...
[INFO] Models loaded successfully!
 * Serving Flask app 'web_app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in production.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Step 8: Access the Web Interface

1. Open your web browser
2. Navigate to: **http://localhost:5000**
3. Click **"Start Detection"**
4. Allow webcam access if prompted
5. You should see your webcam feed with face mask detection!

**First Run Tips**:
- Position yourself 1-2 meters from the camera
- Ensure good lighting
- The system needs 1-2 seconds to initialize detection

---

## Verification

### System Health Check

Run this comprehensive test:

```python
# Create a file: test_installation.py
import sys
print(f"Python version: {sys.version}")

import cv2
print(f"✓ OpenCV {cv2.__version__}")

import numpy as np
print(f"✓ NumPy {np.__version__}")

import tensorflow as tf
print(f"✓ TensorFlow {tf.__version__}")

from flask import Flask
import flask
print(f"✓ Flask {flask.__version__}")

# Test webcam
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Webcam accessible")
    cap.release()
else:
    print("✗ Webcam not accessible")

# Test model loading
from tensorflow.keras.models import load_model
try:
    model = load_model("mask_detector_best.h5")
    print(f"✓ Model loaded (input shape: {model.input_shape})")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

# Test face detector
try:
    faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    print("✓ Face detector loaded")
except Exception as e:
    print(f"✗ Face detector failed: {e}")

print("\n✅ Installation verification complete!")
```

Run the test:
```bash
python test_installation.py
```

All items should show ✓ for a successful installation.

---

## Troubleshooting

### Common Issues

#### Issue 1: "python: command not found"

**Windows**:
- Python not added to PATH during installation
- Solution: Reinstall Python and check "Add Python to PATH"
- Or use `py` instead of `python`: `py web_app.py`

**macOS/Linux**:
- Use `python3` instead of `python`
- Solution: `alias python=python3` (add to ~/.bashrc or ~/.zshrc)

#### Issue 2: "pip: command not found"

```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

#### Issue 3: "Permission denied" errors (Linux/macOS)

```bash
# Don't use sudo with pip
# Instead, use virtual environment (recommended)
# Or install with --user flag
pip install --user -r requirements_web.txt
```

#### Issue 4: Webcam not accessible

**Windows**:
1. Check Camera Privacy Settings:
   - Settings → Privacy → Camera
   - Enable "Allow apps to access your camera"
2. Close other applications using webcam (Skype, Teams, etc.)
3. Try different camera index:
   ```python
   # In web_app.py, line ~35
   camera = cv2.VideoCapture(1)  # Try 0, 1, 2
   ```

**Linux**:
```bash
# Check camera permissions
sudo usermod -a -G video $USER
# Logout and login again

# Test camera
cheese  # Or use webcam test app
```

**macOS**:
- System Preferences → Security & Privacy → Camera
- Allow Terminal/Python to access camera

#### Issue 5: OpenCV installation fails

```bash
# Try installing from conda-forge
conda install -c conda-forge opencv

# Or install headless version (no GUI needed)
pip install opencv-python-headless
```

#### Issue 6: TensorFlow installation fails

**Windows (Python 3.12+)**:
```bash
# TensorFlow may not support latest Python
# Use Python 3.11 instead
# Or install nightly build
pip install tf-nightly
```

**macOS Apple Silicon (M1/M2)**:
```bash
# Use tensorflow-macos
pip install tensorflow-macos tensorflow-metal
```

**Linux (CUDA GPU)**:
```bash
# For GPU support
pip install tensorflow[and-cuda]
```

#### Issue 7: Model file not found

```bash
# Check if files exist
ls -lh *.h5 *.prototxt *.caffemodel

# Verify you're in the correct directory
pwd  # Should be in Project-main/

# If files are missing, download from repository
```

#### Issue 8: Port 5000 already in use

```bash
# Find process using port 5000
# Windows
netstat -ano | findstr :5000

# macOS/Linux
lsof -i :5000

# Kill the process or change port in web_app.py
# Edit line: app.run(debug=True, threaded=True, port=5001)
```

#### Issue 9: "Cannot import name 'escape' from 'jinja2'"

```bash
# Update Flask and Jinja2
pip install --upgrade flask jinja2
```

#### Issue 10: NumPy version conflicts

```bash
# Ensure compatible versions
pip install numpy==2.2.6 opencv-python==4.12.0.88
```

### Getting Help

If issues persist:

1. **Check Logs**: Look for error messages in terminal
2. **GitHub Issues**: Search existing issues or create new one
3. **Stack Overflow**: Tag questions with `flask`, `opencv`, `tensorflow`
4. **Documentation**: Review official docs for specific packages

---

## Alternative Installation Methods

### Method 1: Using Conda (Recommended for Data Science Users)

```bash
# Create conda environment
conda create -n mask-detection python=3.10
conda activate mask-detection

# Install packages
conda install flask opencv tensorflow numpy
pip install -r requirements_web.txt
```

### Method 2: Using Docker (Advanced)

```bash
# Create Dockerfile (example)
docker build -t mask-detection .
docker run -p 5000:5000 --device=/dev/video0 mask-detection
```

### Method 3: Using Pipenv

```bash
# Install pipenv
pip install pipenv

# Create environment and install dependencies
pipenv install
pipenv shell

# Run app
python web_app.py
```

### Method 4: System-wide Installation (Not Recommended)

```bash
# Install without virtual environment
pip install -r requirements_web.txt

# Risk: May conflict with system packages
```

---

## Post-Installation Steps

### 1. Test the System

Run through complete workflow:
1. Start application
2. Access web interface
3. Start detection
4. Verify face detection works
5. Test with/without mask
6. Check statistics update
7. Verify CSV logging
8. Test stop/reset functions

### 2. Optimize for Your Hardware

#### For CPU-only Systems:
```python
# In web_app.py, add at the top:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
```

#### For GPU Systems:
```bash
# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 3. Configure for Production

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Disabling debug mode
- Setting up HTTPS
- Configuring firewall
- Adding authentication
- Performance tuning

---

## Next Steps

After successful installation:

1. ✅ Read [API_DOCUMENTATION.md](API_DOCUMENTATION.md) to understand endpoints
2. ✅ Review [TRAINING_GUIDE.md](TRAINING_GUIDE.md) to improve model accuracy
3. ✅ Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
4. ✅ Explore [WEB_README.md](WEB_README.md) for web interface features

---

## Uninstallation

To remove the project:

```bash
# Deactivate virtual environment
deactivate

# Remove project folder
cd ..
rm -rf Project-main  # Linux/macOS
# Or delete folder manually on Windows

# Remove virtual environment
rm -rf venv
```

---

## Additional Resources

- **Python Installation Guide**: https://www.python.org/downloads/
- **OpenCV Documentation**: https://docs.opencv.org/
- **TensorFlow Installation**: https://www.tensorflow.org/install
- **Flask Documentation**: https://flask.palletsprojects.com/
- **Virtual Environments**: https://docs.python.org/3/tutorial/venv.html

---

## Changelog

- **v1.0.0** (2026-01-12): Initial installation guide
  - Complete installation steps for Windows, macOS, Linux
  - Comprehensive troubleshooting section
  - Alternative installation methods
  - Verification scripts

---

**Need Help?**
- Open an issue on GitHub
- Email: support@example.com
- Check troubleshooting section above

**Last Updated**: January 12, 2026  
**Compatible with**: Project Version 1.0+
