# app.py - Complete Flask Web App for Emotion Detection
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# -------------------------------
# 1. Configuration
# -------------------------------
app = Flask(__name__)
app.secret_key = 'super_secret_key_123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
MODEL_PATH = 'face_emotionModel.h5'
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load emotion labels
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse mapping: number → emotion
EMOTION_MAP = {v: k for k, v in class_indices.items()}
EMOTIONS_LIST = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion messages
EMOTION_MESSAGES = {
    'angry': "You are frowning. Why are you angry?",
    'disgust': "You look disgusted. What's bothering you?",
    'fear': "You look scared. Are you okay?",
    'happy': "You are smiling! Great to see you happy!",
    'neutral': "You look neutral. Everything alright?",
    'sad': "You are frowning. Why are you sad?",
    'surprise': "You look surprised! What's new?"
}

# -------------------------------
# 2. Database Functions
# -------------------------------
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def save_student_data(name, student_id, emotion, image_path):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO students (name, student_id, emotion, image_path) VALUES (?, ?, ?, ?)',
        (name, student_id, emotion, image_path)
    )
    conn.commit()
    conn.close()
    print(f"Saved: {name} - {emotion}")

# -------------------------------
# 3. Prediction Function
# -------------------------------
def predict_emotion(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    emotion_idx = np.argmax(predictions)
    emotion = EMOTION_MAP[emotion_idx]
    confidence = float(predictions[0][emotion_idx]) * 100

    return emotion, confidence

# -------------------------------
# 4. Routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        student_id = request.form['student_id']
        
        # Get uploaded file
        if 'photo' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['photo']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file:
            # Secure filename
            filename = secure_filename(f"{name}_{student_id}_{file.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            
            # Predict emotion
            emotion, confidence = predict_emotion(image_path)
            message = EMOTION_MESSAGES.get(emotion, "Unknown emotion")
            
            # Save to database
            save_student_data(name, student_id, emotion, image_path)
            
            # Pass result to template
            return render_template('index.html', 
                                 result=message,
                                 emotion=emotion,
                                 confidence=f"{confidence:.1f}%",
                                 image_path=image_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)