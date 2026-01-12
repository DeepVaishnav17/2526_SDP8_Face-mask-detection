# Face Mask Detection System - Deployment Guide

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Production Setup](#local-production-setup)
3. [Cloud Deployment Options](#cloud-deployment-options)
4. [Security Hardening](#security-hardening)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Backup and Recovery](#backup-and-recovery)
8. [Scaling Strategies](#scaling-strategies)

---

## Pre-Deployment Checklist

### Essential Tasks

- [ ] **Disable Debug Mode** in Flask
- [ ] **Set Strong Secret Keys**
- [ ] **Configure HTTPS/SSL**
- [ ] **Set up Environment Variables**
- [ ] **Implement Authentication**
- [ ] **Configure CORS Properly**
- [ ] **Set up Error Logging**
- [ ] **Configure Rate Limiting**
- [ ] **Test on Production-like Environment**
- [ ] **Prepare Backup Strategy**
- [ ] **Document Configuration**
- [ ] **Set up Monitoring**

### Code Modifications Required

Before deploying, you must modify [web_app.py](web_app.py):

```python
# ❌ NEVER deploy with these settings:
app.run(debug=True)  # Security risk!

# ✅ Production settings:
app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## Local Production Setup

### Step 1: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-super-secret-key-here-change-this
DATABASE_URL=sqlite:///violations.db
LOG_LEVEL=INFO
MAX_WORKERS=4
CONFIDENCE_THRESHOLD=0.75
```

### Step 2: Create Configuration File

Create `config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = False
    TESTING = False
    
    # Model paths
    FACE_DETECTOR_PROTO = 'deploy.prototxt'
    FACE_DETECTOR_MODEL = 'res10_300x300_ssd_iter_140000.caffemodel'
    MASK_DETECTOR_MODEL = 'mask_detector_best.h5'
    
    # Detection settings
    CONFIDENCE_MIN = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.75))
    SMOOTHING_FRAMES = 10
    FACE_ID_THRESHOLD = 80  # pixels
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'app.log'
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'
    # Additional production settings

class TestingConfig(Config):
    TESTING = True
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

### Step 3: Update web_app.py

Modify the Flask app initialization:

```python
# Add at the top of web_app.py
from config import config
import os
import logging
from logging.handlers import RotatingFileHandler

# Initialize app with config
config_name = os.environ.get('FLASK_ENV', 'production')
app = Flask(__name__)
app.config.from_object(config[config_name])

# Set up logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler(
        'logs/mask_detection.log',
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('Face Mask Detection System startup')

# Replace hardcoded values with config
CONFIDENCE_MIN = app.config['CONFIDENCE_MIN']
```

### Step 4: Use Production WSGI Server

**Install Gunicorn** (Linux/macOS) or **Waitress** (Windows):

```bash
# Linux/macOS
pip install gunicorn

# Windows
pip install waitress
```

**Run with Gunicorn** (Linux/macOS):
```bash
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

**Run with Waitress** (Windows):
```bash
waitress-serve --host=0.0.0.0 --port=5000 web_app:app
```

**Recommended Gunicorn Settings**:
```bash
gunicorn \
  --workers 4 \
  --threads 2 \
  --timeout 120 \
  --bind 0.0.0.0:5000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --log-level info \
  web_app:app
```

### Step 5: Set up Reverse Proxy (Nginx)

Install Nginx:
```bash
# Ubuntu/Debian
sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

Configure Nginx (`/etc/nginx/sites-available/mask-detection`):

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your_cert.crt;
    ssl_certificate_key /etc/ssl/private/your_key.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy settings
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Video feed endpoint (larger buffer)
    location /video_feed {
        proxy_pass http://127.0.0.1:5000;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files
    location /static/ {
        alias /path/to/Project-main/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Increase client max body size (for file uploads)
    client_max_body_size 10M;
}
```

Enable and start Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/mask-detection /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

---

## Cloud Deployment Options

### Option 1: AWS EC2 Deployment

#### Step 1: Launch EC2 Instance

1. **Instance Type**: t3.medium or better (2 vCPU, 4GB RAM)
2. **AMI**: Ubuntu Server 22.04 LTS
3. **Storage**: 20GB minimum
4. **Security Group**:
   - SSH (22) - Your IP only
   - HTTP (80) - 0.0.0.0/0
   - HTTPS (443) - 0.0.0.0/0

#### Step 2: Connect and Setup

```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-venv nginx git

# Clone project
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection/Project-main

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_web.txt
pip install gunicorn
```

#### Step 3: Create Systemd Service

Create `/etc/systemd/system/mask-detection.service`:

```ini
[Unit]
Description=Face Mask Detection System
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/face-mask-detection/Project-main
Environment="PATH=/home/ubuntu/face-mask-detection/Project-main/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/home/ubuntu/face-mask-detection/Project-main/venv/bin/gunicorn \
    --workers 4 \
    --bind 127.0.0.1:5000 \
    --timeout 120 \
    web_app:app

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl start mask-detection
sudo systemctl enable mask-detection
sudo systemctl status mask-detection
```

#### Step 4: Configure SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Option 2: Azure App Service Deployment

#### Step 1: Prepare for Azure

Create `startup.sh`:
```bash
#!/bin/bash
gunicorn --workers 4 --timeout 120 --bind 0.0.0.0:8000 web_app:app
```

Create `requirements.txt` (Azure uses this instead of requirements_web.txt):
```bash
cp requirements_web.txt requirements.txt
echo "gunicorn==21.2.0" >> requirements.txt
```

#### Step 2: Deploy via Azure CLI

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name mask-detection-rg --location eastus

# Create App Service plan
az appservice plan create \
    --name mask-detection-plan \
    --resource-group mask-detection-rg \
    --sku B2 \
    --is-linux

# Create web app
az webapp create \
    --resource-group mask-detection-rg \
    --plan mask-detection-plan \
    --name your-app-name \
    --runtime "PYTHON:3.10" \
    --startup-file startup.sh

# Deploy code
az webapp up \
    --resource-group mask-detection-rg \
    --name your-app-name \
    --runtime "PYTHON:3.10"
```

#### Step 3: Configure App Settings

```bash
az webapp config appsettings set \
    --resource-group mask-detection-rg \
    --name your-app-name \
    --settings \
        FLASK_ENV=production \
        CONFIDENCE_THRESHOLD=0.75 \
        SECRET_KEY="your-secret-key"
```

### Option 3: Google Cloud Run (Serverless)

#### Step 1: Create Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt
RUN pip install gunicorn

# Copy application
COPY . .

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=8080

# Run application
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 0 web_app:app
```

#### Step 2: Deploy to Cloud Run

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project your-project-id

# Build and deploy
gcloud run deploy mask-detection \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

### Option 4: Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt
RUN pip install gunicorn

# Copy application files
COPY web_app.py config.py ./
COPY templates/ templates/
COPY static/ static/
COPY *.h5 *.prototxt *.caffemodel ./

# Create logs directory
RUN mkdir logs

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "web_app:app"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - CONFIDENCE_THRESHOLD=0.75
    volumes:
      - ./logs:/app/logs
      - ./mask_violations.csv:/app/mask_violations.csv
    restart: unless-stopped
    devices:
      - /dev/video0:/dev/video0  # Webcam access

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
    restart: unless-stopped
```

Build and run:
```bash
docker-compose up -d
docker-compose logs -f
```

---

## Security Hardening

### 1. Authentication Implementation

Add Flask-Login for user authentication:

```python
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Implement login logic
    pass

@app.route('/start_detection', methods=['POST'])
@login_required
def start_detection():
    # Now requires authentication
    pass
```

### 2. Rate Limiting

```bash
pip install Flask-Limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

@app.route("/start_detection", methods=["POST"])
@limiter.limit("10 per minute")
def start_detection():
    pass
```

### 3. CORS Configuration

```bash
pip install Flask-CORS
```

```python
from flask_cors import CORS

# Allow specific origins
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

### 4. Input Validation

```python
from flask import request, abort

@app.route('/api/some_endpoint', methods=['POST'])
def some_endpoint():
    # Validate content type
    if not request.is_json:
        abort(415, "Content-Type must be application/json")
    
    # Validate required fields
    data = request.get_json()
    if 'field' not in data:
        abort(400, "Missing required field")
    
    return jsonify({"status": "success"})
```

### 5. Security Headers

```python
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response
```

---

## Performance Optimization

### 1. Model Optimization

```python
# Use TensorFlow Lite for faster inference
import tensorflow as tf

# Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(maskNet)
tflite_model = converter.convert()

# Save model
with open('mask_detector.tflite', 'wb') as f:
    f.write(tflite_model)

# Load and use TFLite model
interpreter = tf.lite.Interpreter(model_path="mask_detector.tflite")
interpreter.allocate_tensors()
```

### 2. GPU Acceleration

```python
import tensorflow as tf

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        app.logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        app.logger.error(f"GPU configuration error: {e}")
```

### 3. Caching

```bash
pip install Flask-Caching
```

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/get_stats')
@cache.cached(timeout=1)  # Cache for 1 second
def get_stats():
    return jsonify(stats)
```

### 4. Frame Processing Optimization

```python
# Process every Nth frame for better performance
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while detection_active:
    ret, frame = camera.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        # Just display without processing
        continue
    
    # Process frame
    detect_and_predict_mask(frame)
```

---

## Monitoring and Logging

### 1. Application Monitoring

```bash
pip install prometheus-flask-exporter
```

```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)

# Custom metrics
detection_counter = metrics.counter(
    'detection_total', 'Total detections',
    labels={'result': lambda: 'mask' if result == 'mask' else 'no_mask'}
)
```

### 2. Health Check Endpoint

```python
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': faceNet is not None and maskNet is not None,
        'camera_available': camera is not None and camera.isOpened() if camera else False
    })
```

### 3. Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler, SMTPHandler

# File handler
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(file_handler)

# Email handler for errors
if not app.debug:
    mail_handler = SMTPHandler(
        mailhost=('smtp.gmail.com', 587),
        fromaddr='noreply@yourdomain.com',
        toaddrs=['admin@yourdomain.com'],
        subject='Mask Detection Error',
        credentials=('user', 'password'),
        secure=()
    )
    mail_handler.setLevel(logging.ERROR)
    app.logger.addHandler(mail_handler)
```

---

## Backup and Recovery

### 1. Database Backup (if using SQLite)

```bash
# Create backup script: backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_FILE="violations.db"

# Create backup
cp $DB_FILE "$BACKUP_DIR/violations_$DATE.db"

# Keep only last 7 days
find $BACKUP_DIR -name "violations_*.db" -mtime +7 -delete
```

Run daily with cron:
```bash
0 2 * * * /path/to/backup.sh
```

### 2. Configuration Backup

```bash
# Backup all config files
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
    config.py \
    .env \
    nginx.conf \
    /etc/systemd/system/mask-detection.service
```

### 3. Model Files Backup

```bash
# Backup models to cloud storage
aws s3 sync . s3://your-bucket/models/ \
    --include "*.h5" \
    --include "*.caffemodel" \
    --include "*.prototxt"
```

---

## Scaling Strategies

### 1. Load Balancing

```nginx
# Nginx load balancer config
upstream mask_detection {
    least_conn;
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    location / {
        proxy_pass http://mask_detection;
    }
}
```

### 2. Redis for Shared State

```bash
pip install redis flask-redis
```

```python
from flask_redis import FlaskRedis

redis_client = FlaskRedis(app)

# Store stats in Redis
redis_client.set('total_detections', stats['total_detections'])
```

### 3. Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mask-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mask-detection
  template:
    metadata:
      labels:
        app: mask-detection
    spec:
      containers:
      - name: mask-detection
        image: your-registry/mask-detection:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: mask-detection-service
spec:
  selector:
    app: mask-detection
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

---

## Post-Deployment Checklist

- [ ] SSL certificate installed and auto-renewal configured
- [ ] Monitoring dashboards set up
- [ ] Backup automation running
- [ ] Log rotation configured
- [ ] Health check endpoints responding
- [ ] Error alerting configured
- [ ] Documentation updated with deployment details
- [ ] Team trained on deployment procedures
- [ ] Disaster recovery plan documented
- [ ] Performance baselines established

---

## Maintenance

### Regular Tasks

**Daily**:
- Check error logs
- Verify backup completion
- Monitor system resources

**Weekly**:
- Review security logs
- Check SSL certificate expiry
- Update dependencies (if needed)

**Monthly**:
- Test disaster recovery
- Review performance metrics
- Update documentation

---

## Support and Resources

- **Production Issues**: Check logs in `/logs/` directory
- **SSL Issues**: Review Let's Encrypt logs
- **Performance**: Use `htop` and `nvidia-smi` (if GPU)
- **Nginx**: Check `/var/log/nginx/error.log`

**Emergency Contacts**:
- System Admin: admin@yourdomain.com
- DevOps Team: devops@yourdomain.com

---

**Last Updated**: January 12, 2026  
**Deployment Version**: 1.0.0
