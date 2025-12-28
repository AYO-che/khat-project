# predict.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "handwriting_efficientnet_final.keras"   
IMG_SIZE = 224
CLASSES = ["beautiful", "medium", "ugly"]       

# ===============================
# PREPROCESS FUNCTION (same logic)
# ===============================
def preprocess_image_safe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Minimal enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Invert only if writing is light
    if np.mean(gray) > 127:
        inverted = gray
    else:
        inverted = 255 - gray

    # Resize keeping aspect ratio
    h, w = inverted.shape
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(inverted, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding
    top = (IMG_SIZE - new_h) // 2
    bottom = IMG_SIZE - new_h - top
    left = (IMG_SIZE - new_w) // 2
    right = IMG_SIZE - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=255
    )

    final = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    final = tf.keras.applications.efficientnet.preprocess_input(final)
    return final


# ===============================
# LOAD MODEL
# ===============================
print("📌 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model Loaded!")


# ===============================
# PREDICT – Single Image
# ===============================
def predict_image(image_path):
    img = preprocess_image_safe(image_path)
    if img is None:
        print("❌ Error loading:", image_path)
        return

    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]

    idx = np.argmax(pred)
    label = CLASSES[idx]
    conf = round(pred[idx] * 100, 2)

    print(f"\n🖼 Image: {image_path}")
    print(f"🎯 Prediction: {label} ({conf}%)")
    print("📊 Probabilities:")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {round(pred[i]*100,2)}%")


# ===============================
# PREDICT – A folder
# ===============================
def predict_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            predict_image(os.path.join(folder_path, file))


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    # change this before running
    TEST_PATH = "testu.jpg"      
    if os.path.isdir(TEST_PATH):
        predict_folder(TEST_PATH)
    else:
        predict_image(TEST_PATH)
