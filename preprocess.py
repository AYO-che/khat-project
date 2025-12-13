import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  
PREPROCESSED_DIR = "dataset_split"
OUTPUT_DIR = "predataset_split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
VAL_RATIO = 0.2  

for folder in [TRAIN_DIR, VAL_DIR]:
    os.makedirs(folder, exist_ok=True)

for subset in ["train", "val"]:
    subset_path = os.path.join(PREPROCESSED_DIR, subset)
    for label in os.listdir(subset_path):
        class_path = os.path.join(subset_path, label)
        if not os.path.isdir(class_path):
            continue

        for folder in [TRAIN_DIR, VAL_DIR]:
            os.makedirs(os.path.join(folder, label), exist_ok=True)

        images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        train_imgs, val_imgs = train_test_split(images, test_size=VAL_RATIO, random_state=42)

        def copy_and_clean(img_list, dest_folder):
            for img_name in tqdm(img_list, desc=f"Processing {label}"):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipped corrupted image: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 8
                )

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                save_path = os.path.join(dest_folder, label, img_name)
                cv2.imwrite(save_path, clean)

        copy_and_clean(train_imgs, TRAIN_DIR)
        copy_and_clean(val_imgs, VAL_DIR)

print(" Preprocessed dataset is now split into train and val folders in grayscale, cleaned!")