import cv2
import mediapipe as mp
import pyttsx3
import math
import time
from playsound import playsound  # For playing MP3 sound

# --- Config ---
MOUTH_OPEN_THRESHOLD = 0.035   # Adjust threshold for sensitivity
CONSEC_FRAMES = 12              # Frames above threshold before triggering alert
SPEAK_MSG = "Yawning detected. Please take a break."
FPS_DISPLAY = True
alert_cooldown = 6  # seconds

# --- Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

engine = pyttsx3.init()
engine.setProperty("rate", 150)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

consec_count = 0
last_alert_time = 0

# Landmark indices for mouth
UPPER_INNER = 13
LOWER_INNER = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291

prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        h, w = frame.shape[:2]
        mouth_ratio = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            def lm(i):
                lm = face_landmarks.landmark[i]
                return (lm.x * w, lm.y * h)

            p_up = lm(UPPER_INNER)
            p_low = lm(LOWER_INNER)
            p_left = lm(LEFT_CORNER)
            p_right = lm(RIGHT_CORNER)

            vert_dist = math.dist(p_up, p_low)
            horz_dist = math.dist(p_left, p_right)

            if horz_dist > 0:
                mouth_ratio = vert_dist / horz_dist

            # Draw visuals
            cv2.circle(frame, (int(p_up[0]), int(p_up[1])), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(p_low[0]), int(p_low[1])), 2, (0, 255, 0), -1)
            cv2.line(frame, (int(p_up[0]), int(p_up[1])), (int(p_low[0]), int(p_low[1])), (0, 255, 0), 1)
            cv2.line(frame, (int(p_left[0]), int(p_left[1])), (int(p_right[0]), int(p_right[1])), (255, 0, 0), 1)
            cv2.putText(frame, f"Ratio: {mouth_ratio:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if mouth_ratio > MOUTH_OPEN_THRESHOLD:
                consec_count += 1
            else:
                consec_count = 0

            if consec_count >= CONSEC_FRAMES and (time.time() - last_alert_time) > alert_cooldown:
                # Voice alert
                engine.say(SPEAK_MSG)
                engine.runAndWait()

                # Play buzzer.mp3
                playsound("buzzer.mp3")

                last_alert_time = time.time()
                cv2.putText(frame, "YAWN ALERT!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if FPS_DISPLAY:
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Yawning Alert", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
