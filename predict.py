import tensorflow as tf
import numpy as np
import cv2
import sys
import os

IMG_SIZE = (224, 224)
MODEL_PATH = "handwriting_mobilenetv2_final.keras"
CLASS_NAMES = ["beautiful", "medium", "ugly"]

model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully")

if len(sys.argv) != 2:
    print("Usage: python predict.py path_to_image")
    sys.exit()

img_path = sys.argv[1]

if not os.path.exists(img_path):
    print(" Image not found")
    sys.exit()

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print(" Cannot read image")
    sys.exit()

if len(img.shape) == 2:  # grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
elif img.shape[2] == 4:  # RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
else:  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(img_lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)
img_lab = cv2.merge((l,a,b))
img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
img = cv2.bitwise_and(img, mask)

img = cv2.resize(img, IMG_SIZE)
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

preds = model.predict(img)[0]
pred_index = np.argmax(preds)
pred_class = CLASS_NAMES[pred_index]
confidence = preds[pred_index]

print("\n🖊 Prediction Result")
print("----------------------")
print(f"Class      : {pred_class}")
print(f"Confidence : {confidence:.2f}")

print("\nFull probabilities:")
for cls, p in zip(CLASS_NAMES, preds):
    print(f"{cls:10s}: {p:.3f}")