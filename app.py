# app.py  |  run with:  flask run
import cv2, threading, numpy as np, tensorflow as tf
from flask import Flask, Response, render_template

MODEL_PATH = "mask_detector_model.h5"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 128
# labels = ["Mask", "No Mask"]

app = Flask(__name__)
cap = cv2.VideoCapture(0)

labels = ["No Mask", "Mask"]

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
            face = np.expand_dims(face / 255.0, 0)
            no_mask, mask = model.predict(face)[0]

            label_id = 1 if mask > no_mask else 0
            label = labels[label_id]
            color = (0, 255, 0) if label_id == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return "<h2>Mask Detector</h2><img src='/video'>"

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)
