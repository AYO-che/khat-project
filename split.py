import os
import shutil
from sklearn.model_selection import train_test_split

DATA_DIR = "DATASET"
OUTPUT_DIR = "dataset_split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
VAL_RATIO = 0.2  # 20% validation


for folder in [TRAIN_DIR, VAL_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

for label in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_path):
        continue

    for folder in [TRAIN_DIR, VAL_DIR]:
        os.makedirs(os.path.join(folder, label), exist_ok=True)

  
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    
    train_imgs, val_imgs = train_test_split(images, test_size=VAL_RATIO, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, label, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, label, img))

print(" Dataset split complete! Train and Val folders are ready.")
