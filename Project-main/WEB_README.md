# Face Mask Detection System - Web Interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_web.txt
```

### 2. Run the Web Application
```bash
python web_app.py
```

### 3. Open Browser
Navigate to: **http://localhost:5000**

## 📁 Project Structure

```
Project-main/
├── web_app.py                    # Flask backend server
├── templates/
│   └── index.html               # Main web interface
├── static/
│   ├── style.css                # Styling
│   └── script.js                # Frontend logic
├── mask_detector.h5             # Trained model (unchanged)
├── deploy.prototxt              # Face detector config (unchanged)
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detector model (unchanged)
├── mask_violations.csv          # Auto-generated log file
└── requirements_web.txt         # Dependencies
```

## ✨ Features

- **Real-time Detection**: Live webcam feed with instant mask detection
- **Modern UI**: Clean, responsive interface with gradient design
- **Statistics Dashboard**: Track total detections, mask/no-mask counts
- **Activity Log**: Monitor recent detection events
- **CSV Logging**: Automatic logging of violations to `mask_violations.csv`
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 Usage

1. Click **"Start Detection"** to activate the webcam
2. The system will detect faces and classify mask status in real-time
3. View statistics and activity in the right panel
4. Click **"Stop Detection"** to turn off the camera
5. Use **"Reset Stats"** to clear all counters

## 🔧 Models Used (Unchanged)

- **Face Detection**: SSD MobileNet (deploy.prototxt + caffemodel)
- **Mask Classification**: Custom trained MobileNetV2 (mask_detector.h5)
- **Accuracy**: 95%+ detection rate
- **Performance**: Real-time processing

## 📊 Statistics Tracking

The system tracks:
- Total number of detections
- Number of "Mask Detected" cases
- Number of "No Mask" warnings
- Current detection status
- Recent activity timeline

## 📝 Logs

All "No Mask" violations are automatically logged to `mask_violations.csv` with:
- Timestamp
- Status
- Confidence percentage

## 🌐 Browser Compatibility

Works best with:
- Google Chrome
- Mozilla Firefox
- Microsoft Edge
- Safari

## 🛠️ Troubleshooting

### Camera not working?
- Check browser permissions for camera access
- Ensure no other application is using the webcam

### Model loading errors?
- Verify all model files are in the Project-main folder
- Check TensorFlow version compatibility

### Port already in use?
Edit `web_app.py` and change the port:
```python
app.run(debug=True, threaded=True, port=5001)  # Change 5000 to 5001
```

## 💡 Notes

- The original detection models are **NOT modified**
- All existing files remain unchanged
- The web interface is an additional feature
- CSV logging continues to work as before
