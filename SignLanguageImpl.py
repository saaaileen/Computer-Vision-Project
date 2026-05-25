import cv2
import joblib
from collections import deque, Counter
from skimage.feature import hog
import mediapipe as mp
import numpy as np

MODEL_PATH = "svm_asl_mp_model2.joblib"
IMG_SIZE = (64, 64)
STABILITY_FRAMES = 7
STABILITY_THRESHOLD = 5

data = joblib.load(MODEL_PATH)
model = data["model"]
le = data["label_encoder"]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

try:
    mp_hands = mp.solutions.hands
except AttributeError:
    from mediapipe.python import solutions as _mps
    mp_hands = _mps.hands

def crop_hand_from_landmarks(frame, hand_landmarks, padding=20):
    """Crop hand region using MediaPipe landmarks"""
    h, w = frame.shape[:2]
    xs = [int(pt.x * w) for pt in hand_landmarks.landmark]
    ys = [int(pt.y * h) for pt in hand_landmarks.landmark]
    x1, x2 = max(min(xs) - padding, 0), min(max(xs) + padding, w)
    y1, y2 = max(min(ys) - padding, 0), min(max(ys) + padding, h)
    return frame[y1:y2, x1:x2]

# def predict_from_frame_mediapipe(frame, hand_landmarks=None):
#     """Predict ASL sign from frame. If hand_landmarks provided, use them."""
#     if hand_landmarks is None:
#         return None
    
#     crop = crop_hand_from_landmarks(frame, hand_landmarks)
#     if crop is None or crop.size == 0:
#         return None
    
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(gray, IMG_SIZE)
#     features = hog(img,
#         orientations=9,
#         pixels_per_cell=(8, 8),
#         cells_per_block=(2, 2),
#         block_norm='L2-Hys')
#     img = features.reshape(1, -1)
#     pred = model.predict(img)
#     return le.inverse_transform(pred)[0]

def predict_from_frame_mediapipe(hand_landmarks):

    features = []

    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    features = np.array(features)

    # normalize
    features = features - features.mean()
    features = features / (features.std() + 1e-6)

    features = features.reshape(1, -1)

    pred = model.predict(features)

    return le.inverse_transform(pred)[0]

def run_webcam_mediapipe(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            label = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Predict using the actual detected landmarks
                    label = predict_from_frame_mediapipe(hand_landmarks)
            
            disp = label if label is not None else "No hand"
            cv2.putText(frame, disp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ASL SVM (MediaPipe)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
    cap.release()
    cv2.destroyAllWindows()
    
run_webcam_mediapipe()