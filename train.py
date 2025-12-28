import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -----------------------------
# Seeds for reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Paths & constants
# -----------------------------
TRAIN_DIR = "fresh_preprocessed_aug_safe/train"
VAL_DIR   = "fresh_preprocessed_aug_safe/val"  # <-- صححت المسار
IMG_SIZE = (224,224)
BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS = 30
FINE_TUNE_LAYERS = 20

# -----------------------------
# Data augmentation
# -----------------------------
from tensorflow.keras.applications.efficientnet import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.7,1.3],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, seed=SEED
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

class_names = list(train_data.class_indices.keys())
print("Classes:", class_names)

# -----------------------------
# Class weights
# -----------------------------
labels = train_data.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# Build model
# -----------------------------
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(base.input, output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Learning rate scheduler
# -----------------------------
def cosine_annealing(epoch, lr):
    import math
    total_epochs = EPOCHS
    return 1e-4 * (math.cos(math.pi * epoch / total_epochs) + 1) / 2

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    ModelCheckpoint("efficientnet_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    LearningRateScheduler(cosine_annealing)
]

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# Fine-tune last layers
# -----------------------------
base.trainable = True
for layer in base.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_ft = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# Save final model
# -----------------------------
model.save("handwriting_efficientnet_final.keras")
print("\n✅ FINAL model saved as 'handwriting_efficientnet_final.keras'")

# -----------------------------
# Plot training & validation accuracy
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# -----------------------------
# Confusion matrix & classification report
# -----------------------------
val_data.reset()  # <-- صحيح هنا
preds = model.predict(val_data, steps=len(val_data), verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=val_data.class_indices.keys(),
            yticklabels=val_data.class_indices.keys(), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))
