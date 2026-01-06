from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
from flask_cors import CORS
CORS(app)  # يسمح للـ frontend يتصل

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "handwriting_efficientnet_final.keras"
UPLOAD_FOLDER = "uploads"
IMG_SIZE = 224
import json

with open("classes.json", "r") as f:
    class_indices = json.load(f)

CLASSES = {v: k for k, v in class_indices.items()}


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ===============================
# PREPROCESS (نفس predict.py)
# ===============================
def preprocess_image_safe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)

    if np.mean(gray) <= 127:
        gray = 255 - gray

    h, w = gray.shape
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(gray, (new_w, new_h))

    top = (IMG_SIZE - new_h)//2
    bottom = IMG_SIZE - new_h - top
    left = (IMG_SIZE - new_w)//2
    right = IMG_SIZE - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=255
    )

    final = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    final = tf.keras.applications.efficientnet.preprocess_input(final)
    return final

# ===============================
# API ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    print("Received request /predict")  # هذا باش تشوف فالـ terminal
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = preprocess_image_safe(path)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]

    result = {
    "label": CLASSES[int(np.argmax(preds))],
    "confidence": round(float(np.max(preds)) * 100, 2),
    "probabilities": {
        CLASSES[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(CLASSES))
    }
}


    return jsonify(result)


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
