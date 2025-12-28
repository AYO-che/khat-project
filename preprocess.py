import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RAW_DIR = "dataset_split/train"
OUTPUT_DIR = "fresh_preprocessed_aug_safe"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
VAL_RATIO = 0.2
IMG_SIZE = 224

# Create output folders
for folder in [TRAIN_DIR, VAL_DIR]:
    os.makedirs(folder, exist_ok=True)

# Safe preprocessing function: minimal changes to preserve writing
def preprocess_image_safe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Keep original colors but convert to grayscale for simplicity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Only minimal enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Invert only if writing is light
    if np.mean(gray) > 127:  # likely white background, dark writing
        inverted = gray
    else:
        inverted = 255 - gray  # make writing black

    # Resize while keeping aspect ratio
    h, w = inverted.shape
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(inverted, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding to IMG_SIZE x IMG_SIZE
    top = (IMG_SIZE - new_h) // 2
    bottom = IMG_SIZE - new_h - top
    left = (IMG_SIZE - new_w) // 2
    right = IMG_SIZE - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=255)
    final = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    return final

# Data augmentation setup (light)
datagen = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.9,1.1],
    horizontal_flip=False
)

# Process dataset
for label in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, label)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📂 Class: {label}")
    os.makedirs(os.path.join(TRAIN_DIR, label), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, label), exist_ok=True)

    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(images) == 0:
        print(f"⚠️ Empty class skipped: {label}")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=VAL_RATIO, shuffle=True, random_state=42)

    def work(img_list, dest_folder, augment=True):
        for img_name in tqdm(img_list, desc=f"Processing {label}"):
            src = os.path.join(class_path, img_name)
            processed = preprocess_image_safe(src)
            if processed is None:
                print(f"❌ Corrupted image skipped: {src}")
                continue
            save_path = os.path.join(dest_folder, label, img_name)
            cv2.imwrite(save_path, processed)

            # Augmentation for train set
            if augment:
                img_array = np.expand_dims(processed, 0)
                aug_iter = datagen.flow(img_array, batch_size=1,
                                        save_to_dir=os.path.join(dest_folder, label),
                                        save_prefix='aug', save_format='png')
                for _ in range(3):
                    next(aug_iter)

    work(train_imgs, TRAIN_DIR, augment=True)
    work(val_imgs, VAL_DIR, augment=False)

print("\n✅ DONE! Preprocessed + Augmented dataset created in:", OUTPUT_DIR)
