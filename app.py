import io
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp

MODEL_PATH = "svm_asl_mp_model.joblib"

data = joblib.load(MODEL_PATH)
model = data["model"]
le = data["label_encoder"]

# mediapipe helpers (compat)
try:
    mp_hands = mp.solutions.hands
except AttributeError:
    from mediapipe.python import solutions as _mps
    mp_hands = _mps.hands

def predict_from_landmarks(hand_landmarks):
    feats = []
    for lm in hand_landmarks.landmark:
        feats.extend([lm.x, lm.y, lm.z])
    feats = np.array(feats)
    feats = feats - feats.mean()
    feats = feats / (feats.std() + 1e-6)
    feats = feats.reshape(1, -1)
    pred = model.predict(feats)
    return le.inverse_transform(pred)[0]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400
    file = request.files["image"]
    data_bytes = file.read()
    nparr = np.frombuffer(data_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    # MediaPipe expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True,
                        model_complexity=0,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({"labels": [], "message": "No hand detected"})

    labels = []
    for hand_landmarks in results.multi_hand_landmarks:
        label = predict_from_landmarks(hand_landmarks)
        labels.append(label)

    return jsonify({"labels": labels})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)