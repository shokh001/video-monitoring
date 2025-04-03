from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# YOLOv8 modelini yuklash
model = YOLO("yolov8n.pt")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamera ochish
cap = cv2.VideoCapture(0)
video_width, video_height = 640, 480

# Ogohlantirishlar ro‘yxati
warnings = []

def detect_head_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            left_eye_x = left_eye.x * video_width
            right_eye_x = right_eye.x * video_width
            nose_tip_x = nose_tip.x * video_width
            nose_tip_y = nose_tip.y * video_height
            
            eye_center_x = (left_eye_x + right_eye_x) / 2
            dx = nose_tip_x - eye_center_x
            dy = nose_tip_y - (left_eye.y * video_height + right_eye.y * video_height) / 2
            yaw_angle = np.arctan2(dx, dy) * 180 / np.pi

            if yaw_angle > 90:
                yaw_angle -= 180
            elif yaw_angle < -90:
                yaw_angle += 180
            
            return yaw_angle
    
    return 0

def generate_frames():
    global warnings
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (video_width, video_height))
        head_turn_angle = detect_head_pose(frame)
        detections = model(frame)

        people_count, phone_detected = 0, False
        
        for box in detections[0].boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            if class_name == "person":
                people_count += 1
            if class_name == "cell phone":
                phone_detected = True
        
        warnings = []
        if people_count > 1:
            warnings.append("👥 Imtihonda faqat bitta odam qatnashishi mumkin❗")
        if phone_detected:
            warnings.append("📱 Qurilmalardan foydalanish taqiqlangan❗")
        if people_count == 1 and abs(head_turn_angle) > 30:
            warnings.append(f"🕵 Ekranga qarang, yordam olishga harakat qilmang {abs(head_turn_angle):.2f}°❗")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warnings')
def get_warnings():
    return jsonify(warnings)

if __name__ == '__main__':
    app.run(debug=True)
