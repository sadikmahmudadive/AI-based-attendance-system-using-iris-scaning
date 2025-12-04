"""
Retrain Iris Recognition Model with New Users
==============================================
This script allows you to add new users to the iris recognition system
and retrain the model to recognize them.

Usage:
1. Capture iris images for new users
2. Run this script to retrain the model
3. The updated model will recognize the new users

Author: Iris Attendance System
"""

import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
import json

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'CASIA-Iris-Interval'
NEW_USERS_DIR = BASE_DIR / 'new_users'
MODELS_DIR = BASE_DIR / 'models'
IMG_SIZE = 224
MIN_IMAGES_PER_USER = 3  # Minimum images needed per user

# Create directories
NEW_USERS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_and_crop_eye(image):
    """Detect and crop eye region from image."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Try face detection first
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    best_eye = None
    best_size = 0
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            
            for (ex, ey, ew, eh) in eyes:
                if ew * eh > best_size:
                    best_size = ew * eh
                    pad = int(ew * 0.4)
                    ex1, ey1 = max(0, ex-pad), max(0, ey-pad)
                    ex2 = min(roi_color.shape[1], ex+ew+pad)
                    ey2 = min(roi_color.shape[0], ey+eh+pad)
                    best_eye = roi_color[ey1:ey2, ex1:ex2]
    
    # Try direct eye detection
    if best_eye is None:
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes:
            if ew * eh > best_size:
                best_size = ew * eh
                pad = int(ew * 0.4)
                ex1, ey1 = max(0, ex-pad), max(0, ey-pad)
                ex2 = min(image.shape[1], ex+ew+pad)
                ey2 = min(image.shape[0], ey+eh+pad)
                best_eye = image[ey1:ey2, ex1:ex2]
    
    return best_eye


def capture_iris_images(user_id, user_name, num_images=5):
    """Capture iris images from webcam for a new user."""
    print(f"\n{'='*60}")
    print(f"📸 CAPTURING IRIS IMAGES FOR: {user_name} (ID: {user_id})")
    print(f"{'='*60}")
    
    user_dir = NEW_USERS_DIR / user_id
    user_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    captured = 0
    print(f"\n📷 Position your eye close to the camera")
    print(f"   Press SPACE to capture, Q to quit")
    print(f"   Need {num_images} images, captured: 0/{num_images}\n")
    
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Detect eyes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # Draw rectangles around eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add status text
        cv2.putText(display, f"User: {user_name} | Captured: {captured}/{num_images}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE: Capture | Q: Quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if len(eyes) > 0:
            cv2.putText(display, "Eye Detected!", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Capture Iris', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            if len(eyes) > 0:
                # Crop the largest eye
                largest_eye = max(eyes, key=lambda e: e[2] * e[3])
                x, y, w, h = largest_eye
                pad = int(w * 0.3)
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2 = min(frame.shape[1], x+w+pad)
                y2 = min(frame.shape[0], y+h+pad)
                eye_img = frame[y1:y2, x1:x2]
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{user_id}_{captured+1}_{timestamp}.jpg"
                filepath = user_dir / filename
                cv2.imwrite(str(filepath), eye_img)
                
                captured += 1
                print(f"   ✅ Captured image {captured}/{num_images}")
            else:
                print("   ⚠️ No eye detected! Position your eye in the frame.")
        
        elif key == ord('q'):
            print("\n⚠️ Capture cancelled by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured >= MIN_IMAGES_PER_USER:
        print(f"\n✅ Successfully captured {captured} images for {user_name}")
        return True
    else:
        print(f"\n❌ Need at least {MIN_IMAGES_PER_USER} images. Only captured {captured}.")
        return False


def add_images_from_folder(user_id, source_folder):
    """Add existing images to a new user's folder."""
    user_dir = NEW_USERS_DIR / user_id
    user_dir.mkdir(exist_ok=True)
    
    source = Path(source_folder)
    if not source.exists():
        print(f"❌ Source folder not found: {source_folder}")
        return False
    
    count = 0
    for img_file in source.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Try to crop eye
            eye_crop = detect_and_crop_eye(img)
            if eye_crop is not None:
                img = eye_crop
            
            # Save
            dest = user_dir / f"{user_id}_{count+1}{img_file.suffix}"
            cv2.imwrite(str(dest), img)
            count += 1
    
    print(f"✅ Added {count} images for user {user_id}")
    return count >= MIN_IMAGES_PER_USER


def load_dataset():
    """Load all images from CASIA dataset and new users."""
    images = []
    labels = []
    
    print("\n📂 Loading dataset...")
    
    # Load CASIA dataset
    if DATASET_DIR.exists():
        for person_dir in sorted(DATASET_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_id = person_dir.name
            person_images = []
            
            for eye_dir in person_dir.iterdir():
                if eye_dir.is_dir():
                    for img_file in eye_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                                person_images.append(img)
            
            if len(person_images) >= MIN_IMAGES_PER_USER:
                for img in person_images:
                    images.append(img)
                    labels.append(person_id)
        
        print(f"   CASIA dataset: {len(set(labels))} users loaded")
    
    # Load new users
    new_user_count = 0
    if NEW_USERS_DIR.exists():
        for user_dir in NEW_USERS_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            
            user_id = user_dir.name
            user_images = []
            
            for img_file in user_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        user_images.append(img)
            
            if len(user_images) >= MIN_IMAGES_PER_USER:
                for img in user_images:
                    images.append(img)
                    labels.append(user_id)
                new_user_count += 1
    
    print(f"   New users: {new_user_count} users loaded")
    print(f"   Total: {len(set(labels))} users, {len(images)} images")
    
    return np.array(images), np.array(labels)


def create_model(num_classes):
    """Create EfficientNetV2 model for iris recognition."""
    # Load base model
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model


def retrain_model():
    """Retrain the model with all users (CASIA + new users)."""
    print("\n" + "="*60)
    print("🔄 RETRAINING IRIS RECOGNITION MODEL")
    print("="*60)
    
    # Load dataset
    X, y = load_dataset()
    
    if len(X) == 0:
        print("❌ No images found!")
        return False
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total images: {len(X)}")
    print(f"   Total classes: {num_classes}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Create model
    print("\n🏗️ Building model...")
    model, base_model = create_model(num_classes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            str(MODELS_DIR / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Phase 1: Train head
    print("\n📈 Phase 1: Training classification head...")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print("\n📈 Phase 2: Fine-tuning entire model...")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"   Validation Accuracy: {val_acc:.2%}")
    
    # Save model and label encoder
    print("\n💾 Saving model...")
    model.save(MODELS_DIR / 'efficientnetv2_attendance.keras')
    joblib.dump(le, MODELS_DIR / 'le_master.joblib')
    
    # Save training info
    info = {
        'trained_at': datetime.now().isoformat(),
        'num_classes': num_classes,
        'classes': le.classes_.tolist(),
        'val_accuracy': float(val_acc),
        'total_images': len(X)
    }
    with open(MODELS_DIR / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Model retrained and saved successfully!")
    print(f"   Classes: {num_classes}")
    print(f"   Accuracy: {val_acc:.2%}")
    
    return True


def list_new_users():
    """List all new users added."""
    print("\n📋 New Users:")
    print("-" * 40)
    
    if not NEW_USERS_DIR.exists():
        print("   No new users directory found")
        return
    
    for user_dir in sorted(NEW_USERS_DIR.iterdir()):
        if user_dir.is_dir():
            img_count = len(list(user_dir.glob('*')))
            status = "✅" if img_count >= MIN_IMAGES_PER_USER else "⚠️"
            print(f"   {status} {user_dir.name}: {img_count} images")
    
    print("-" * 40)


def main():
    """Main menu for retraining system."""
    print("\n" + "="*60)
    print("🎯 IRIS RECOGNITION - ADD NEW USERS & RETRAIN")
    print("="*60)
    
    while True:
        print("\n📋 MENU:")
        print("   1. Capture iris images from webcam (new user)")
        print("   2. Add images from folder (new user)")
        print("   3. List new users")
        print("   4. Retrain model with all users")
        print("   5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            user_id = input("Enter User ID (e.g., EMP001): ").strip()
            user_name = input("Enter User Name: ").strip()
            if user_id and user_name:
                num = int(input("Number of images to capture (default 5): ").strip() or "5")
                capture_iris_images(user_id, user_name, num)
            else:
                print("❌ User ID and Name are required")
        
        elif choice == '2':
            user_id = input("Enter User ID (e.g., EMP001): ").strip()
            folder = input("Enter path to folder with iris images: ").strip()
            if user_id and folder:
                add_images_from_folder(user_id, folder)
            else:
                print("❌ User ID and folder path are required")
        
        elif choice == '3':
            list_new_users()
        
        elif choice == '4':
            confirm = input("⚠️ This will retrain the model. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                retrain_model()
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    main()
