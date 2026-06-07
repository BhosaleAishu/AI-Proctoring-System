import cv2
import mediapipe as mp
import time
import sys

# -------- INITIALIZE --------
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    sys.exit()

# -------- VARIABLES --------
warning_count = 0
last_warning_time = 0
warning_delay = 2
screenshot_taken = False

print("Press ESC or Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()

    # -------- FACE DETECTION --------
    results_face = face_detection.process(rgb)
    face_count = 0
    face_box = None

    if results_face.detections:
        for detection in results_face.detections:
            face_count += 1
            mp_drawing.draw_detection(frame, detection)

            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face_box = (x, y, width, height)

    # -------- FACE CONDITIONS --------
    if face_count == 0:
        cv2.putText(frame, "No Face Detected!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    elif face_count > 1:
        cv2.putText(frame, "Multiple Faces Detected!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    else:
        cv2.putText(frame, "Single Candidate", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # -------- EYE TRACKING (FIXED PROPERLY) --------
    results_mesh = face_mesh.process(rgb)

    if results_mesh.multi_face_landmarks and face_box:
        for face_landmarks in results_mesh.multi_face_landmarks:

            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            lx = int(left_eye.x * w)
            rx = int(right_eye.x * w)

            cv2.circle(frame, (lx, int(left_eye.y*h)), 3, (255,0,0), -1)
            cv2.circle(frame, (rx, int(right_eye.y*h)), 3, (255,0,0), -1)

            eye_center = (lx + rx) / 2

            # -------- RELATIVE TO FACE --------
            x, y, fw, fh = face_box

            relative_position = (eye_center - x) / fw  # normalized (0 to 1)

            # -------- CORRECT LOGIC --------
            if relative_position < 0.4:
                text = "Looking Left"
                color = (0,0,255)

                if current_time - last_warning_time > warning_delay:
                    warning_count += 1
                    last_warning_time = current_time

            elif relative_position > 0.6:
                text = "Looking Right"
                color = (0,0,255)

                if current_time - last_warning_time > warning_delay:
                    warning_count += 1
                    last_warning_time = current_time

            else:
                text = "Looking Center"
                color = (0,255,0)

            cv2.putText(frame, text, (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

    # -------- WARNINGS --------
    cv2.putText(frame, f"Warnings: {warning_count}", (50,150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)

    # -------- SCREENSHOT --------
    if warning_count > 3 and not screenshot_taken:
        filename = f"cheating_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        screenshot_taken = True

    if screenshot_taken:
        cv2.putText(frame, "Screenshot Captured!", (50,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),2)

    # -------- DISPLAY --------
    cv2.imshow("AI Proctoring System", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 27 or key == ord('q'):
        break

# -------- CLEAN EXIT --------
cap.release()
cv2.destroyAllWindows()
sys.exit()