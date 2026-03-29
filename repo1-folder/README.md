# 🛡️ FaceGuard v2 — Face Mask Detection System

## What It Does

| Camera sees | Result |
|---|---|
| 😷 Person with mask | Shows "MASK ON" — no identity check, NOT logged |
| ✅ Registered person, no mask | Shows their name — logs photo + timestamp to Word doc |
| ❓ Unknown person, no mask | Shows "UNKNOWN" |
| 👤 No face | Shows "No face detected" |

---

## Requirements

- Python 3.9+
- Node.js 18+
- pip, npm

**No CMake, no dlib, no C++ compiler needed.**

---

## Install & Run

### Windows
```
Double-click start.bat
```

### Mac / Linux
```bash
chmod +x start.sh && ./start.sh
```

### Manual
```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm start
```

Then open: **http://localhost:3000** in Chrome

---

## First Run Note

DeepFace downloads the **Facenet512** model (~250MB) and MediaPipe downloads face mesh model (~10MB) on first use. After that, they load from cache instantly.

---

## How to Use

### 1. Register a person
- Go to **Register** tab
- Type their name
- Look into camera → click Capture Photo
- Click Register Person

### 2. Run detection
- Go to **Detection** tab
- Click **Start Detection**
- Camera sends a frame every 2 seconds automatically

### 3. View logs
- Go to **Logs** tab
- See all identified entries
- Click **Download Word Doc** to get `entry_log.docx`

---

## Tech Stack

| Layer | Tech | Why |
|---|---|---|
| Face recognition | DeepFace + Facenet512 | No dlib/cmake, very accurate, persists via JSON |
| Mask detection | MediaPipe FaceMesh | 468 landmarks, runs offline, no CNN needed |
| Backend | FastAPI | Fast, async, clean |
| Frontend | React + react-webcam | Webcam capture, live detection |
| Word log | python-docx | Photo + name + timestamp per entry |

---

## Troubleshooting

**"No face detected" when registering?**
- Face the camera directly
- Ensure good lighting (not dark, no strong backlight)
- Keep your face within the green guide box
- Do NOT wear a mask while registering

**First run slow?**
- Normal — models are downloading (~260MB total, once)
- Wait for backend terminal to show "Application startup complete"

**Port 8000 already in use?**
```bash
# Kill it
# Windows: netstat -ano | findstr :8000  then  taskkill /PID <pid> /F
# Mac/Linux: lsof -ti:8000 | xargs kill
```

**Camera not working?**
- Use Chrome or Edge (best webcam support)
- Allow camera permissions when prompted
- Make sure no other app is using the camera
