# train_mask_detector.py
# ----------------------
import os, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------------------------------------------------ #
# CONFIG
IMG_SIZE   = 128        # 128×128 RGB
BATCH      = 32
EPOCHS     = 15
DATA_DIR   = "dataset"  # dataset/with_mask , dataset/without_mask
MODEL_PATH = "mask_detector_model.h5"
# ------------------------------------------------------------------ #

# 1️⃣  data pipeline (with on‑the‑fly augmentation for robustness)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 2️⃣  model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    BatchNormalization(), MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'), BatchNormalization(),
    MaxPooling2D(), Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'), BatchNormalization(),
    MaxPooling2D(), Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3️⃣  training
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"✅  Model saved to {MODEL_PATH}")
