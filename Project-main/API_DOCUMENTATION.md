# Face Mask Detection API Documentation

## Overview

This document provides comprehensive API documentation for the Face Mask Detection Web Application. The application provides real-time face mask detection with multi-face tracking capabilities.

**Base URL**: `http://localhost:5000`  
**API Version**: 1.0  
**Response Format**: JSON  

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [WebSocket Streams](#websocket-streams)
6. [Rate Limiting](#rate-limiting)

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API Key authentication
- JWT tokens
- OAuth 2.0

---

## Endpoints

### 1. Home Page

```
GET /
```

**Description**: Renders the main web interface

**Response**: HTML page

**Example**:
```bash
curl http://localhost:5000/
```

---

### 2. Start Detection

```
POST /start_detection
```

**Description**: Starts the face mask detection system and activates the webcam

**Request**: No body required

**Response**:
```json
{
  "status": "success",
  "message": "Detection started"
}
```

**Status Codes**:
- `200 OK`: Detection started successfully
- `500 Internal Server Error`: Camera initialization failed

**Example**:
```bash
curl -X POST http://localhost:5000/start_detection
```

**Notes**:
- Initializes webcam (cv2.VideoCapture(0))
- Loads face detection and mask classification models
- Starts real-time processing thread

---

### 3. Stop Detection

```
POST /stop_detection
```

**Description**: Stops the detection system and releases the webcam

**Request**: No body required

**Response**:
```json
{
  "status": "success",
  "message": "Detection stopped"
}
```

**Status Codes**:
- `200 OK`: Detection stopped successfully

**Example**:
```bash
curl -X POST http://localhost:5000/stop_detection
```

**Notes**:
- Releases camera resources
- Clears per-face tracking buffers
- Maintains statistics counters

---

### 4. Get Statistics

```
GET /get_stats
```

**Description**: Retrieves current detection statistics and activity log

**Response**:
```json
{
  "total_detections": 127,
  "mask_count": 95,
  "no_mask_count": 32,
  "current_status": "Active",
  "activity": [
    {
      "time": "14:32:15",
      "message": "ID2: Mask detected",
      "type": "success"
    },
    {
      "time": "14:32:10",
      "message": "ID1: No mask detected!",
      "type": "warning"
    }
  ]
}
```

**Fields**:
- `total_detections` (integer): Total number of unique face detections
- `mask_count` (integer): Number of faces detected wearing masks
- `no_mask_count` (integer): Number of faces detected without masks
- `current_status` (string): "Active" or "Inactive"
- `activity` (array): Last 10 detection events

**Activity Event Object**:
- `time` (string): Timestamp in HH:MM:SS format
- `message` (string): Event description with face ID
- `type` (string): "success" (mask) or "warning" (no mask)

**Example**:
```bash
curl http://localhost:5000/get_stats
```

**Notes**:
- Thread-safe statistics access using locks
- Activity log is limited to last 10 events
- Statistics persist across start/stop cycles until reset

---

### 5. Reset Statistics

```
POST /reset_stats
```

**Description**: Resets all detection statistics and activity log to zero

**Request**: No body required

**Response**:
```json
{
  "status": "success",
  "message": "Statistics reset"
}
```

**Status Codes**:
- `200 OK`: Statistics reset successfully

**Example**:
```bash
curl -X POST http://localhost:5000/reset_stats
```

**Notes**:
- Resets counters to zero
- Clears activity log
- Does not affect detection state (active/inactive)

---

### 6. Video Feed (MJPEG Stream)

```
GET /video_feed
```

**Description**: Provides real-time video stream with face mask detection overlay

**Response**: Multipart MJPEG stream

**Content-Type**: `multipart/x-mixed-replace; boundary=frame`

**Example**:
```html
<img src="/video_feed" alt="Video Feed">
```

**Stream Format**:
```
--frame
Content-Type: image/jpeg

[JPEG Binary Data]
--frame
Content-Type: image/jpeg

[JPEG Binary Data]
...
```

**Overlay Features**:
- Bounding boxes around detected faces
  - Green box: Mask detected
  - Red box: No mask detected
- Face ID labels (ID0, ID1, ID2...)
- Confidence percentage
- Label text ("Mask" or "No Mask")

**Notes**:
- Automatically updates when detection is active
- Shows blank frame when detection is stopped
- Frame rate: ~15-30 FPS depending on hardware
- Resolution: 640x480 (default webcam resolution)

---

## Data Models

### Statistics Object

```typescript
interface Statistics {
  total_detections: number;    // Total unique face detections
  mask_count: number;           // Faces with masks
  no_mask_count: number;        // Faces without masks
  current_status: string;       // "Active" | "Inactive"
  activity: ActivityEvent[];    // Last 10 events
}
```

### Activity Event Object

```typescript
interface ActivityEvent {
  time: string;      // Format: "HH:MM:SS"
  message: string;   // "ID{n}: Mask detected" | "ID{n}: No mask detected!"
  type: string;      // "success" | "warning"
}
```

### API Response Object

```typescript
interface ApiResponse {
  status: string;    // "success" | "error"
  message: string;   // Human-readable message
  data?: any;        // Optional response data
}
```

---

## Error Handling

All API endpoints follow consistent error response format:

```json
{
  "status": "error",
  "message": "Detailed error description",
  "code": "ERROR_CODE"
}
```

**Common Error Codes**:

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `CAMERA_INIT_FAILED` | Unable to access webcam | 500 |
| `MODEL_LOAD_FAILED` | Unable to load AI models | 500 |
| `INVALID_REQUEST` | Malformed request | 400 |
| `DETECTION_NOT_ACTIVE` | Operation requires active detection | 409 |

**Example Error Response**:
```json
{
  "status": "error",
  "message": "Unable to access webcam. Check if another application is using the camera.",
  "code": "CAMERA_INIT_FAILED"
}
```

---

## Technical Details

### Detection Algorithm

1. **Face Detection**: Uses SSD MobileNet (OpenCV DNN)
   - Model: `res10_300x300_ssd_iter_140000.caffemodel`
   - Config: `deploy.prototxt`
   - Confidence threshold: 0.5

2. **Mask Classification**: Custom MobileNetV2 (TensorFlow/Keras)
   - Model: `mask_detector_best.h5`
   - Accuracy: 98.25%
   - Input size: 224x224 RGB
   - Confidence threshold: 0.75

3. **Multi-Face Tracking**:
   - Position-based face ID assignment
   - 80px proximity threshold for ID persistence
   - Independent 10-frame smoothing per face
   - Automatic cleanup of inactive face IDs

### Performance Specifications

- **Latency**: ~30-50ms per frame (depending on number of faces)
- **Throughput**: 15-30 FPS on typical laptop hardware
- **Max Concurrent Faces**: Tested up to 10 faces simultaneously
- **Memory Usage**: ~500MB (models + video buffer)
- **CPU Usage**: 25-40% on quad-core processor

### Violation Logging

All "No Mask" detections are logged to `mask_violations.csv`:

```csv
Timestamp,Face_ID,Status,Confidence
2026-01-12 14:32:15,ID1,No Mask,92.5%
2026-01-12 14:32:18,ID3,No Mask,88.3%
```

**CSV Fields**:
- `Timestamp`: Detection time (YYYY-MM-DD HH:MM:SS)
- `Face_ID`: Unique face identifier
- `Status`: Always "No Mask" (only violations logged)
- `Confidence`: Prediction confidence percentage

---

## OpenAPI Specification

```yaml
openapi: 3.0.0
info:
  title: Face Mask Detection API
  version: 1.0.0
  description: Real-time face mask detection system with multi-face tracking
  contact:
    name: API Support
    email: support@example.com

servers:
  - url: http://localhost:5000
    description: Development server

paths:
  /start_detection:
    post:
      summary: Start detection system
      description: Initializes webcam and starts real-time face mask detection
      responses:
        '200':
          description: Detection started successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: success
                  message:
                    type: string
                    example: Detection started
        '500':
          description: Camera initialization failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /stop_detection:
    post:
      summary: Stop detection system
      description: Releases webcam and stops detection processing
      responses:
        '200':
          description: Detection stopped successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: success
                  message:
                    type: string
                    example: Detection stopped

  /get_stats:
    get:
      summary: Get detection statistics
      description: Retrieves current statistics and activity log
      responses:
        '200':
          description: Statistics retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Statistics'

  /reset_stats:
    post:
      summary: Reset statistics
      description: Resets all counters and activity log
      responses:
        '200':
          description: Statistics reset successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: success
                  message:
                    type: string
                    example: Statistics reset

  /video_feed:
    get:
      summary: Video stream
      description: MJPEG stream with detection overlay
      responses:
        '200':
          description: Video stream
          content:
            multipart/x-mixed-replace:
              schema:
                type: string
                format: binary

components:
  schemas:
    Statistics:
      type: object
      properties:
        total_detections:
          type: integer
          example: 127
        mask_count:
          type: integer
          example: 95
        no_mask_count:
          type: integer
          example: 32
        current_status:
          type: string
          enum: [Active, Inactive]
          example: Active
        activity:
          type: array
          items:
            $ref: '#/components/schemas/ActivityEvent'
    
    ActivityEvent:
      type: object
      properties:
        time:
          type: string
          example: "14:32:15"
        message:
          type: string
          example: "ID2: Mask detected"
        type:
          type: string
          enum: [success, warning]
          example: success
    
    Error:
      type: object
      properties:
        status:
          type: string
          example: error
        message:
          type: string
          example: Error description
        code:
          type: string
          example: ERROR_CODE
```

---

## Testing the API

### Using cURL

```bash
# Start detection
curl -X POST http://localhost:5000/start_detection

# Get statistics
curl http://localhost:5000/get_stats

# Stop detection
curl -X POST http://localhost:5000/stop_detection

# Reset statistics
curl -X POST http://localhost:5000/reset_stats
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:5000"

# Start detection
response = requests.post(f"{BASE_URL}/start_detection")
print(response.json())

# Get statistics
response = requests.get(f"{BASE_URL}/get_stats")
stats = response.json()
print(f"Total detections: {stats['total_detections']}")
print(f"Mask count: {stats['mask_count']}")

# Stop detection
response = requests.post(f"{BASE_URL}/stop_detection")
print(response.json())
```

### Using JavaScript (Fetch API)

```javascript
// Start detection
fetch('http://localhost:5000/start_detection', { method: 'POST' })
  .then(res => res.json())
  .then(data => console.log(data));

// Get statistics
fetch('http://localhost:5000/get_stats')
  .then(res => res.json())
  .then(stats => {
    console.log('Total detections:', stats.total_detections);
    console.log('Mask count:', stats.mask_count);
  });
```

---

## Rate Limiting Recommendations

For production deployment, implement rate limiting:

```python
# Example with Flask-Limiter
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/start_detection", methods=["POST"])
@limiter.limit("10 per minute")
def start_detection():
    # Implementation
    pass
```

---

## WebSocket Support (Future Enhancement)

For real-time statistics updates without polling:

```javascript
// Future WebSocket implementation
const ws = new WebSocket('ws://localhost:5000/ws/stats');

ws.onmessage = function(event) {
    const stats = JSON.parse(event.data);
    updateDashboard(stats);
};
```

---

## CORS Configuration

For cross-origin requests in production:

```python
from flask_cors import CORS

# Allow specific origins
CORS(app, resources={
    r"/api/*": {"origins": ["https://yourdomain.com"]}
})
```

---

## Additional Resources

- **Project Repository**: [GitHub Link]
- **Training Guide**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Installation Guide**: See [INSTALLATION.md](INSTALLATION.md)
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Model Performance**: See [classification_report.txt](classification_report.txt)

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Email: support@example.com
- Documentation: This file

**Last Updated**: January 12, 2026  
**API Version**: 1.0.0
