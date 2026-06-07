import face_recognition
import os

def authenticate_face(frame):
    known_dir = "known_faces"

    rgb = frame[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb)

    if len(encodings) == 0:
        return False

    user_encoding = encodings[0]

    for file in os.listdir(known_dir):
        img = face_recognition.load_image_file(f"{known_dir}/{file}")
        known_encoding = face_recognition.face_encodings(img)[0]

        match = face_recognition.compare_faces([known_encoding], user_encoding)

        if match[0]:
            return True

    return False