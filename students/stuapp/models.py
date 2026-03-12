import os
from django.db import models as django_models
from django.conf import settings

class Stuapp(django_models.Model):
    title = django_models.CharField(max_length=150)
    amount = django_models.DecimalField(decimal_places=2, max_digits=10)
    description = django_models.TextField(blank=True)

    def __str__(self):
        return self.title

try:
    import tensorflow as tf
except ImportError:
    tf = None

import numpy as np
import cv2

# Use absolute path for the model
MODEL_DIR = os.path.join(settings.BASE_DIR, "DL")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model (4).h5")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model = None
if tf and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model not found at {MODEL_PATH}")

CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N',
    'O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

def predict_sign(image_path):
    if model is None:
        return "Model not loaded"
    
    try:
        # Read image from file path
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not read image"
        
        # Resize and convert to grayscale (model expects 100x100x1)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = np.expand_dims(img, axis=[0, -1])  # Add batch and channel dimensions

        prediction = model.predict(img, verbose=0)
        class_index = np.argmax(prediction)

        print(f"Prediction probabilities: {prediction}")
        print(f"Predicted class index: {class_index}")
        print(f"Available classes: {len(CLASS_NAMES)}")

        if class_index >= len(CLASS_NAMES):
            return "Unknown"

        return CLASS_NAMES[class_index]
    except Exception as e:
        return f"Error: {str(e)}"
