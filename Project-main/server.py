from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime, timedelta
import threading
import os
import base64
import cv2
import numpy as np

app = Flask(__name__)
DB_NAME = "mask_history.db"
DATASET_DIR = "dataset"

# Global variable to store the currently active user (from Frontend)
current_user = None
user_lock = threading.Lock()

def init_db():
    """Initialize the SQLite database and create tables."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # History table (Raw logs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Attendance Log (Processed logs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Initialize DB on startup
init_db()
os.makedirs(DATASET_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_page')
def register_page():
    return render_template('register_person.html')

@app.route('/set_user', methods=['POST'])
def set_user():
    """Endpoint for Frontend to set the active user."""
    global current_user
    data = request.json
    with user_lock:
        current_user = data.get('name')
    return jsonify({"status": "success", "current_user": current_user})

@app.route('/get_user', methods=['GET'])
def get_user():
    """Endpoint for Python Script to check who is the active user."""
    with user_lock:
        return jsonify({"current_user": current_user})

@app.route('/register_user', methods=['POST'])
def register_user():
    """Register a new user and save their face image."""
    try:
        name = request.form.get('name')
        image_file = request.files.get('image')

        if not name or not image_file:
            return jsonify({"error": "Name and image are required"}), 400

        # Save User to DB
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                conn.commit()
            except sqlite3.IntegrityError:
                pass # User exists, just adding more photos

        # Save Image to Dataset
        user_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(user_dir, exist_ok=True)
        
        # Unique filename using timestamp
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(user_dir, filename)
        image_file.save(save_path)

        return jsonify({"status": "success", "message": f"User {name} registered."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/record_status', methods=['POST'])
def record_status():
    """Endpoint for Python Script to save mask status/attendance."""
    data = request.json
    name = data.get('name')
    status = data.get('status')
    
    if not name or not status:
        return jsonify({"error": "Missing name or status"}), 400

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # 1. Save to History (Raw high-frequency)
            cursor.execute("INSERT INTO history (name, status) VALUES (?, ?)", (name, status))
            
            # 2. Save to Attendance Log (Throttled/More meaningful)
            # Find user_id
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            user_row = cursor.fetchone()
            user_id = user_row[0] if user_row else None
            
            cursor.execute("INSERT INTO attendance_log (user_id, name, status) VALUES (?, ?, ?)", 
                           (user_id, name, status))
            conn.commit()
            
            # Auto-cleanup: Delete raw history older than 7 days
            seven_days_ago = datetime.now() - timedelta(days=7)
            cursor.execute("DELETE FROM history WHERE timestamp < ?", (seven_days_ago,))
            conn.commit()
            
        return jsonify({"status": "recorded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history/<name>', methods=['GET'])
def get_history(name):
    """Endpoint to get last 7 days history for a specific person."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            seven_days_ago = datetime.now() - timedelta(days=7)
            cursor.execute("SELECT name, status, timestamp FROM attendance_log WHERE name = ? AND timestamp > ? ORDER BY timestamp DESC", (name, seven_days_ago))
            rows = cursor.fetchall()
            
        history = [{"name": r[0], "status": r[1], "timestamp": r[2]} for r in rows]
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/attendance', methods=['GET'])
def get_all_attendance():
    """Endpoint to get all attendance logs."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, status, timestamp FROM attendance_log ORDER BY timestamp DESC LIMIT 100")
            rows = cursor.fetchall()
            
        history = [{"name": r[0], "status": r[1], "timestamp": r[2]} for r in rows]
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# CORS support (Basic)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    print("Starting Attendance Backend on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
