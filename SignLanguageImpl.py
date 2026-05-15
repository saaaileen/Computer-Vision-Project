import cv2
import joblib
from collections import deque, Counter

MODEL_PATH = r"E:\Kuliah\comvision\projek\svm_asl_model.joblib"
IMG_SIZE = (64, 64)
STABILITY_FRAMES = 7   # number of frames to consider
STABILITY_THRESHOLD = 5  # min votes out of STABILITY_FRAMES to accept

data = joblib.load(MODEL_PATH)
model = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]

import mediapipe as mp

mp_hands = mp.solutions.hands

def crop_hand_mediapipe(frame, padding=20):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        xs = [int(pt.x * w) for pt in lm.landmark]
        ys = [int(pt.y * h) for pt in lm.landmark]
        x1, x2 = max(min(xs) - padding, 0), min(max(xs) + padding, w)
        y1, y2 = max(min(ys) - padding, 0), min(max(ys) + padding, h)
        return frame[y1:y2, x1:x2]

def predict_from_frame_mediapipe(frame):
    crop = crop_hand_mediapipe(frame)
    if crop is None:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, IMG_SIZE).flatten().reshape(1, -1)
    img_s = scaler.transform(img)
    pred = model.predict(img_s)
    return le.inverse_transform(pred)[0]

def run_webcam_mediapipe(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label = predict_from_frame_mediapipe(frame)
        disp = label if label is not None else "No hand"
        cv2.putText(frame, disp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("ASL SVM (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()