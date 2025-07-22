import os
import time
import cv2
import numpy as np
import tensorflow as tf
import winsound

MODEL_PATH = "mask_detector_model.h5"
CASCADE    = "haarcascade_frontalface_default.xml"
IMG_SIZE   = 128
ALERT_WAV  = os.path.abspath("alert.wav")
THRESHOLD  = 0.5

# Time in seconds to wait before next alert
ALERT_COOLDOWN = 2  # 2 seconds
last_alert_time = 0

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE)

labels = ["No Mask", "Mask"]
colors = [(0, 0, 255), (0, 255, 0)]

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "[ERROR] Could not access webcam."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prob_no_mask, prob_mask = model.predict(face)[0]
        label_id = 0 if prob_no_mask > prob_mask else 1
        confidence = max(prob_no_mask, prob_mask)

        # Draw label and box
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[label_id], 2)
        cv2.putText(frame, f"{labels[label_id]}: {confidence*100:.1f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[label_id], 2)

        # Alert logic
        current_time = time.time()
        if label_id == 0 and confidence > THRESHOLD:
            if current_time - last_alert_time >= ALERT_COOLDOWN:
                try:
                    winsound.PlaySound(ALERT_WAV,
                                       winsound.SND_FILENAME |
                                       winsound.SND_ASYNC |
                                       winsound.SND_NODEFAULT)
                    last_alert_time = current_time
                except Exception as e:
                    print(f"[WARN] Could not play alert: {e}")

    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
