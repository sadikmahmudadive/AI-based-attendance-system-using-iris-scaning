"""
Iris-Based Attendance System - Web Application
===============================================
A Flask web application for iris-based attendance management.

Features:
- Mark attendance via iris image upload
- View attendance reports
- Enroll new users
- Dashboard with statistics
"""

import os
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import cv2
import tensorflow as tf
import joblib
from werkzeug.utils import secure_filename

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.secret_key = 'iris-attendance-secret-key-2024'

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
UPLOAD_DIR = BASE_DIR / 'uploads'
ATTENDANCE_LOG_DIR = MODELS_DIR / 'attendance_logs'

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
ATTENDANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.25  # Lowered for testing with non-CASIA images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'bmp'}

# ============================================================================
# LOAD MODEL AND LABEL ENCODER
# ============================================================================

print("🔄 Loading Iris Recognition Model...")

# Import preprocess_input for custom objects
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_preprocess_input

# Load model with custom objects
MODEL_PATH = MODELS_DIR / 'efficientnetv2_attendance.keras'
LABEL_ENCODER_PATH = MODELS_DIR / 'le_master.joblib'
ENROLLMENT_DB_PATH = MODELS_DIR / 'enrollment_database.json'

custom_objects = {'preprocess_input': eff_preprocess_input}

try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print(f"✅ Label encoder loaded: {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f"❌ Error loading label encoder: {e}")
    label_encoder = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_enrollment_db():
    """Load enrollment database."""
    if ENROLLMENT_DB_PATH.exists():
        with open(ENROLLMENT_DB_PATH, 'r') as f:
            return json.load(f)
    return {"enrollments": {}, "metadata": {"created": datetime.now().isoformat()}}

def save_enrollment_db(data):
    """Save enrollment database."""
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(ENROLLMENT_DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def get_person_name(iris_id):
    """Get person name from enrollment database."""
    db = load_enrollment_db()
    person = db["enrollments"].get(iris_id, {})
    return person.get("name", f"Unknown ({iris_id})")

def detect_and_crop_eye(image):
    """Detect and crop eye region from image using Haar cascades."""
    # Load Haar cascades for face and eye detection
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    # Convert to grayscale for detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Try to detect faces first
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    best_eye = None
    best_eye_size = 0
    
    if len(faces) > 0:
        # Search for eyes within face regions
        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = image[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            for (ex, ey, ew, eh) in eyes:
                # Get the largest eye detected
                if ew * eh > best_eye_size:
                    best_eye_size = ew * eh
                    # Add padding around the eye
                    pad = int(ew * 0.3)
                    ex1 = max(0, ex - pad)
                    ey1 = max(0, ey - pad)
                    ex2 = min(face_roi_color.shape[1], ex + ew + pad)
                    ey2 = min(face_roi_color.shape[0], ey + eh + pad)
                    best_eye = face_roi_color[ey1:ey2, ex1:ex2]
    
    # If no face detected, try detecting eyes directly
    if best_eye is None:
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in eyes:
            if ew * eh > best_eye_size:
                best_eye_size = ew * eh
                pad = int(ew * 0.3)
                ex1 = max(0, ex - pad)
                ey1 = max(0, ey - pad)
                ex2 = min(image.shape[1], ex + ew + pad)
                ey2 = min(image.shape[0], ey + eh + pad)
                best_eye = image[ey1:ey2, ex1:ex2]
    
    return best_eye

def preprocess_image(image):
    """Preprocess image for model prediction."""
    # Try to detect and crop eye region first
    eye_crop = detect_and_crop_eye(image)
    if eye_crop is not None and eye_crop.size > 0:
        image = eye_crop
        print("👁️ Eye region detected and cropped")
    else:
        print("⚠️ No eye detected, using full image")
    
    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Ensure 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    return np.expand_dims(image, axis=0)

def recognize_iris(image):
    """Recognize iris from image."""
    if model is None or label_encoder is None:
        return None, 0.0, []
    
    processed = preprocess_image(image)
    predictions = model(processed, training=False).numpy()[0]
    
    # Get top-3 predictions
    top_k_indices = np.argsort(predictions)[::-1][:3]
    top_k = [(label_encoder.classes_[idx], float(predictions[idx])) for idx in top_k_indices]
    
    best_idx = top_k_indices[0]
    predicted_id = label_encoder.classes_[best_idx]
    confidence = float(predictions[best_idx])
    
    return predicted_id, confidence, top_k

def get_today_log_path():
    """Get today's attendance log path."""
    today = datetime.now().strftime("%Y-%m-%d")
    return ATTENDANCE_LOG_DIR / f"attendance_{today}.csv"

def ensure_log_file():
    """Ensure today's log file exists with headers."""
    log_path = get_today_log_path()
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "iris_id", "name", "department",
                "employee_id", "confidence", "status", "entry_type"
            ])
    return log_path

def is_duplicate_entry(iris_id, window_minutes=30):
    """Check for duplicate entry within time window."""
    log_path = get_today_log_path()
    if not log_path.exists():
        return False
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["iris_id"] == iris_id:
                last_time = datetime.fromisoformat(row["timestamp"])
                if (datetime.now() - last_time).total_seconds() < window_minutes * 60:
                    return True
    return False

def get_entry_type(iris_id):
    """Determine CHECK_IN or CHECK_OUT."""
    log_path = get_today_log_path()
    if not log_path.exists():
        return "CHECK_IN"
    
    last_entry_type = None
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["iris_id"] == iris_id:
                last_entry_type = row["entry_type"]
    
    return "CHECK_OUT" if last_entry_type == "CHECK_IN" else "CHECK_IN"

def get_attendance_status():
    """Get attendance status based on current time."""
    now = datetime.now()
    work_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    late_threshold = work_start + timedelta(minutes=15)
    
    if now < work_start:
        return "EARLY"
    elif now <= late_threshold:
        return "ON_TIME"
    else:
        return "LATE"

def log_attendance(iris_id, confidence):
    """Log attendance entry."""
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "success": False,
            "message": f"Confidence too low ({confidence:.1%}). Please try again."
        }
    
    if is_duplicate_entry(iris_id):
        return {
            "success": False,
            "message": "Duplicate entry. Please wait 30 minutes."
        }
    
    db = load_enrollment_db()
    person = db["enrollments"].get(iris_id, {})
    name = person.get("name", f"Unknown ({iris_id})")
    department = person.get("department", "")
    employee_id = person.get("employee_id", "")
    
    status = get_attendance_status()
    entry_type = get_entry_type(iris_id)
    timestamp = datetime.now().isoformat()
    
    log_path = ensure_log_file()
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, iris_id, name, department,
            employee_id, f"{confidence:.4f}", status, entry_type
        ])
    
    return {
        "success": True,
        "message": f"{entry_type} recorded successfully!",
        "iris_id": iris_id,
        "name": name,
        "timestamp": timestamp,
        "status": status,
        "entry_type": entry_type,
        "confidence": confidence
    }

def get_attendance_report(date=None):
    """Get attendance report for a date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    log_path = ATTENDANCE_LOG_DIR / f"attendance_{date}.csv"
    if not log_path.exists():
        return []
    
    records = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

def get_attendance_summary(date=None):
    """Get attendance summary statistics."""
    records = get_attendance_report(date)
    if not records:
        return {"total": 0, "on_time": 0, "late": 0, "early": 0, "total_entries": 0}
    
    check_ins = [r for r in records if r["entry_type"] == "CHECK_IN"]
    unique_people = set(r["iris_id"] for r in check_ins)
    
    return {
        "total": len(unique_people),
        "on_time": len(set(r["iris_id"] for r in check_ins if r["status"] == "ON_TIME")),
        "late": len(set(r["iris_id"] for r in check_ins if r["status"] == "LATE")),
        "early": len(set(r["iris_id"] for r in check_ins if r["status"] == "EARLY")),
        "total_entries": len(records)
    }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with dashboard."""
    summary = get_attendance_summary()
    return render_template('index.html', summary=summary, date=datetime.now().strftime("%Y-%m-%d"))

@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    """Mark attendance page."""
    if request.method == 'POST':
        if 'iris_image' not in request.files:
            flash('No image file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['iris_image']
        if file.filename == '':
            flash('No image selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Read and process image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                flash('Could not read image', 'error')
                return redirect(request.url)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Recognize iris
            iris_id, confidence, top_k = recognize_iris(image)
            
            if iris_id is None:
                flash('Model not loaded. Please check server logs.', 'error')
                return redirect(request.url)
            
            # Log attendance
            result = log_attendance(iris_id, confidence)
            
            if result["success"]:
                flash(f'✅ {result["message"]} - {result["name"]} ({result["entry_type"]})', 'success')
            else:
                flash(f'❌ {result["message"]}', 'error')
            
            return render_template('mark_attendance.html', result=result, top_k=top_k)
        else:
            flash('Invalid file type. Allowed: png, jpg, jpeg, tif, bmp', 'error')
    
    return render_template('mark_attendance.html', result=None, top_k=None)

@app.route('/attendance_report')
def attendance_report():
    """View attendance report."""
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    records = get_attendance_report(date)
    summary = get_attendance_summary(date)
    return render_template('attendance_report.html', records=records, summary=summary, date=date)

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    """Enroll new person."""
    detected_iris = None
    confidence = None
    top_predictions = None
    
    if request.method == 'POST':
        # Check if this is an image upload for iris detection
        if 'iris_image' in request.files and request.files['iris_image'].filename != '':
            file = request.files['iris_image']
            
            if file and allowed_file(file.filename):
                # Read and process image
                file_bytes = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Recognize iris
                    iris_id, conf, top_k = recognize_iris(image)
                    
                    if iris_id is not None:
                        detected_iris = iris_id
                        confidence = conf
                        top_predictions = top_k
                        flash(f'🔍 Detected Iris ID: {iris_id} (Confidence: {conf:.1%})', 'info')
                    else:
                        flash('❌ Could not recognize iris. Model not loaded.', 'error')
                else:
                    flash('❌ Could not read the image file.', 'error')
            else:
                flash('❌ Invalid file type. Allowed: png, jpg, jpeg, tif, bmp', 'error')
        
        # Check if this is a form submission for enrollment
        elif 'name' in request.form:
            iris_id = request.form.get('iris_id', '').strip()
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            department = request.form.get('department', '').strip()
            employee_id = request.form.get('employee_id', '').strip()
            phone = request.form.get('phone', '').strip()
            
            if not iris_id or not name:
                flash('Iris ID and Name are required', 'error')
                return redirect(request.url)
            
            db = load_enrollment_db()
            
            # Check if already enrolled
            if iris_id in db["enrollments"]:
                flash(f'⚠️ Iris ID {iris_id} is already enrolled!', 'warning')
                return redirect(request.url)
            
            db["enrollments"][iris_id] = {
                "name": name,
                "email": email,
                "department": department,
                "employee_id": employee_id,
                "phone": phone,
                "enrolled_at": datetime.now().isoformat(),
                "is_active": True
            }
            save_enrollment_db(db)
            
            flash(f'✅ Successfully enrolled {name} (ID: {iris_id})', 'success')
            return redirect(url_for('enrolled_list'))
    
    # Get available iris IDs from label encoder
    available_ids = []
    if label_encoder is not None:
        db = load_enrollment_db()
        enrolled_ids = set(db["enrollments"].keys())
        available_ids = [id for id in label_encoder.classes_ if id not in enrolled_ids]
    
    return render_template('enroll.html', 
                           available_ids=available_ids, 
                           detected_iris=detected_iris,
                           confidence=confidence,
                           top_predictions=top_predictions)

@app.route('/enrolled_list')
def enrolled_list():
    """View enrolled persons."""
    db = load_enrollment_db()
    enrollments = [{"iris_id": k, **v} for k, v in db["enrollments"].items() if v.get("is_active", True)]
    return render_template('enrolled_list.html', enrollments=enrollments)

@app.route('/delete_user/<iris_id>', methods=['POST'])
def delete_user(iris_id):
    """Delete an enrolled user."""
    db = load_enrollment_db()
    
    if iris_id in db["enrollments"]:
        user_name = db["enrollments"][iris_id].get("name", iris_id)
        del db["enrollments"][iris_id]
        save_enrollment_db(db)
        flash(f'🗑️ Successfully deleted {user_name} (ID: {iris_id})', 'success')
    else:
        flash(f'❌ User with ID {iris_id} not found', 'error')
    
    return redirect(url_for('enrolled_list'))

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API endpoint for iris recognition."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if file and allowed_file(file.filename):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Could not read image"}), 400
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        iris_id, confidence, top_k = recognize_iris(image)
        
        if iris_id is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        result = log_attendance(iris_id, confidence)
        result["top_k"] = top_k
        
        return jsonify(result)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/summary')
def api_summary():
    """API endpoint for attendance summary."""
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    summary = get_attendance_summary(date)
    return jsonify(summary)

# ============================================================================
# NEW USER REGISTRATION (FOR RETRAINING)
# ============================================================================

NEW_USERS_DIR = BASE_DIR / 'new_users'
NEW_USERS_DIR.mkdir(exist_ok=True)
MIN_IMAGES_PER_USER = 3

def get_new_users_info():
    """Get list of new users pending for model retraining."""
    users = []
    if NEW_USERS_DIR.exists():
        for user_dir in sorted(NEW_USERS_DIR.iterdir()):
            if user_dir.is_dir():
                img_count = len([f for f in user_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                users.append({
                    'user_id': user_dir.name,
                    'image_count': img_count,
                    'ready': img_count >= MIN_IMAGES_PER_USER
                })
    return users

@app.route('/add_new_user', methods=['GET', 'POST'])
def add_new_user():
    """Add a completely new user to the system (requires retraining)."""
    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip()
        user_name = request.form.get('user_name', '').strip()
        email = request.form.get('email', '').strip()
        department = request.form.get('department', '').strip()
        
        if not user_id or not user_name:
            flash('❌ User ID and Name are required', 'error')
            return redirect(request.url)
        
        # Check if user_id already exists in trained model
        if label_encoder is not None and user_id in label_encoder.classes_:
            flash(f'⚠️ User ID {user_id} already exists in the model', 'warning')
            return redirect(request.url)
        
        # Create user directory
        user_dir = NEW_USERS_DIR / user_id
        user_dir.mkdir(exist_ok=True)
        
        # Save user info
        user_info = {
            'name': user_name,
            'email': email,
            'department': department,
            'created_at': datetime.now().isoformat()
        }
        with open(user_dir / 'info.json', 'w') as f:
            json.dump(user_info, f, indent=2)
        
        # Handle image uploads
        images = request.files.getlist('iris_images')
        saved_count = 0
        
        for i, img_file in enumerate(images):
            if img_file and img_file.filename and allowed_file(img_file.filename):
                file_bytes = np.frombuffer(img_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Try to detect and crop eye
                    eye_crop = detect_and_crop_eye(image)
                    if eye_crop is not None and eye_crop.size > 0:
                        image = eye_crop
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{user_id}_{saved_count+1}_{timestamp}.jpg"
                    cv2.imwrite(str(user_dir / filename), image)
                    saved_count += 1
        
        if saved_count >= MIN_IMAGES_PER_USER:
            flash(f'✅ Added {user_name} with {saved_count} images. Run retraining to activate.', 'success')
        elif saved_count > 0:
            flash(f'⚠️ Added {saved_count} images for {user_name}. Need at least {MIN_IMAGES_PER_USER} images.', 'warning')
        else:
            flash(f'❌ No valid images uploaded for {user_name}', 'error')
        
        return redirect(url_for('add_new_user'))
    
    # Get pending new users
    new_users = get_new_users_info()
    
    return render_template('add_new_user.html', 
                          new_users=new_users, 
                          min_images=MIN_IMAGES_PER_USER)

@app.route('/upload_more_images/<user_id>', methods=['POST'])
def upload_more_images(user_id):
    """Upload additional images for a new user."""
    user_dir = NEW_USERS_DIR / user_id
    
    if not user_dir.exists():
        flash(f'❌ User {user_id} not found', 'error')
        return redirect(url_for('add_new_user'))
    
    images = request.files.getlist('iris_images')
    saved_count = 0
    
    for img_file in images:
        if img_file and img_file.filename and allowed_file(img_file.filename):
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is not None:
                eye_crop = detect_and_crop_eye(image)
                if eye_crop is not None and eye_crop.size > 0:
                    image = eye_crop
                
                existing = len(list(user_dir.glob('*.jpg')))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{user_id}_{existing+1}_{timestamp}.jpg"
                cv2.imwrite(str(user_dir / filename), image)
                saved_count += 1
    
    if saved_count > 0:
        flash(f'✅ Added {saved_count} more images for {user_id}', 'success')
    else:
        flash('❌ No valid images uploaded', 'error')
    
    return redirect(url_for('add_new_user'))

@app.route('/delete_new_user/<user_id>', methods=['POST'])
def delete_new_user(user_id):
    """Delete a pending new user."""
    import shutil
    user_dir = NEW_USERS_DIR / user_id
    
    if user_dir.exists():
        shutil.rmtree(user_dir)
        flash(f'🗑️ Deleted pending user {user_id}', 'success')
    else:
        flash(f'❌ User {user_id} not found', 'error')
    
    return redirect(url_for('add_new_user'))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 IRIS ATTENDANCE SYSTEM - WEB SERVER")
    print("="*60)
    print(f"   Model loaded: {model is not None}")
    print(f"   Classes: {len(label_encoder.classes_) if label_encoder else 0}")
    print("="*60)
    print("\n🌐 Starting server at http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
