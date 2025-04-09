from flask import Flask, render_template, Response, jsonify, request, session
import cv2
import time
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import logging

app = Flask(__name__)
app.secret_key = 'shox'

# Logging sozlamalari
logging.basicConfig(level=logging.DEBUG)

model = YOLO("yolov8n.pt")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480

class ProctorState:
    def __init__(self):
        self.warnings = []
        self.warnings_count = 0
        self.last_warning_time = 0
        self.warning_cooldown = 1  # Tezroq tekshirish uchun 1 sekundga qisqartirdim
        self.active_warnings = set()
        self.test_results = None

    def update_warnings(self, new_warnings):
        current_time = time.time()
        new_warnings_set = set(new_warnings)
        
        if new_warnings and (current_time - self.last_warning_time) > self.warning_cooldown:
            for warning in new_warnings_set:
                if warning not in self.active_warnings:
                    self.warnings_count += 1
                    logging.debug(f"Yangi ogohlantirish qo‘shildi: {warning}, Hisob: {self.warnings_count}")
            
            self.active_warnings = new_warnings_set
            self.warnings = new_warnings
            self.last_warning_time = current_time
        elif not new_warnings:
            self.active_warnings = set()
            self.warnings = []
            logging.debug("Ogohlantirishlar yo‘q, faol ogohlantirishlar tozalandi.")

state = ProctorState()

def detect_head_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            eye_center_x = (left_eye.x + right_eye.x) * VIDEO_WIDTH / 2
            nose_tip_x = nose_tip.x * VIDEO_WIDTH
            nose_tip_y = nose_tip.y * VIDEO_HEIGHT
            eye_center_y = (left_eye.y + right_eye.y) * VIDEO_HEIGHT / 2
            
            dx = nose_tip_x - eye_center_x
            dy = nose_tip_y - eye_center_y
            yaw_angle = np.arctan2(dx, dy) * 180 / np.pi

            logging.debug(f"Bosh burilish burchagi: {yaw_angle:.2f}")
            return np.clip(yaw_angle, -90, 90)
    
    logging.debug("Yuz aniqlanmadi, 0 qaytarildi")
    return 0

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            logging.error("Kameradan rasm olinmadi")
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        
        head_turn_angle = detect_head_pose(frame)
        detections = model(frame)

        people_count, phone_detected = 0, False
        
        # YOLO aniqlash natijalarini tekshirish
        for box in detections[0].boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]
            class_name = model.names[cls]
            logging.debug(f"Aniqlangan ob'ekt: {class_name}, Ishonch: {confidence:.2f}")
            
            if class_name == "person" and confidence > 0.5:
                people_count += 1
            elif class_name == "cell phone" and confidence > 0.5:
                phone_detected = True
                logging.debug("Telefon aniqlandi!")
        
        temp_warnings = []
        
        if people_count > 1:
            temp_warnings.append("👥 Imtihonda faqat bitta odam qatnashishi mumkin❗")
        if phone_detected:
            temp_warnings.append("📱 Qurilmalardan foydalanish taqiqlangan❗")
            logging.debug("Telefon ogohlantirishi qo‘shildi")
        if people_count == 1 and abs(head_turn_angle) > 15:
            temp_warnings.append(f"🕵 Ekranga qarang {abs(head_turn_angle):.2f}°❗")
        
        logging.debug(f"Odamlar soni: {people_count}, Telefon aniqlandi: {phone_detected}, Bosh burchagi: {head_turn_angle:.2f}")
        logging.debug(f"Vaqtinchalik ogohlantirishlar: {temp_warnings}")
        
        state.update_warnings(temp_warnings)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test')
def test():
    state.warnings_count = 0
    state.warnings = []
    return render_template('index.html')

@app.route('/submit-test', methods=['POST'])
def submit_test():
    data = request.json
    state.test_results = data
    return jsonify({"status": "success"})

@app.route('/get-results')
def get_results():
    return jsonify(state.test_results or {})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warnings')
def get_warnings():
    logging.debug(f"Frontendga yuborilgan ogohlantirishlar: {state.warnings}, Hisob: {state.warnings_count}")
    return jsonify({"warnings": state.warnings, "count": state.warnings_count})

def cleanup():
    cap.release()
    face_mesh.close()

if __name__ == '__main__':
    # Model sinflarini tekshirish
    logging.info(f"YOLO model sinflari: {model.names}")
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        cleanup()