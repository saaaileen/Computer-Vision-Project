import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Setup FastAPI App
app = FastAPI(title="SignSense ASL AI", description="Real-time American Sign Language Recognition API")

# Enable CORS for external testing or complex frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Paths
MODEL_PATH = "svm_asl_mp_model.joblib"

# Load the trained Model and Label Encoder
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found in current directory.", file=sys.stderr)
    sys.exit(1)

try:
    print(f"Loading SVM model from '{MODEL_PATH}'...")
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    le = data["label_encoder"]
    classes = list(le.classes_)
    print(f"Model loaded successfully! Supported classes ({len(classes)}): {classes[:10]}... {classes[-10:]}")
except Exception as e:
    print(f"Failed to load the model: {e}", file=sys.stderr)
    sys.exit(1)

# Request validation schema using Pydantic
class Landmark(BaseModel):
    x: float
    y: float
    z: float

class PredictionRequest(BaseModel):
    landmarks: List[Landmark]

class DummyLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class DummyHandLandmarks:
    def __init__(self, landmarks: List[Landmark]):
        self.landmark = [DummyLandmark(lm.x, lm.y, lm.z) for lm in landmarks]

def predict_from_landmarks(hand_landmarks_obj):
    """
    Replicates the prediction logic from SignLanguageImpl.py:
    1. Extracts [x, y, z] features.
    2. Subtracts the mean.
    3. Normalizes by dividing by the standard deviation.
    4. Reshapes and predicts via the SVM model.
    """
    features = []
    for lm in hand_landmarks_obj.landmark:
        features.extend([lm.x, lm.y, lm.z])
    
    features = np.array(features)
    
    # Normalize features exactly as done in training
    mean_val = features.mean()
    std_val = features.std()
    features = features - mean_val
    features = features / (std_val + 1e-6)
    
    features = features.reshape(1, -1)
    
    pred = model.predict(features)
    raw_label = le.inverse_transform(pred)[0]
    
    # Handle numpy type serialization safely (NumPy 2.0 compatibility)
    if isinstance(raw_label, bytes):
        return raw_label.decode("utf-8")
    elif hasattr(np, "bytes_") and isinstance(raw_label, np.bytes_):
        return raw_label.decode("utf-8")
    return str(raw_label)

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        if len(req.landmarks) != 21:
            raise HTTPException(status_code=400, detail="Must provide exactly 21 landmarks.")
        
        hand_landmarks_obj = DummyHandLandmarks(req.landmarks)
        label = predict_from_landmarks(hand_landmarks_obj)
        
        return {
            "status": "success",
            "prediction": label
        }
    except Exception as e:
        import traceback
        print("Prediction exception caught:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Prediction failed: {str(e)}"}
        )

# Mount static files (will hold css/ and js/)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            "<h1>Frontend Template Missing</h1><p>Please wait for the server to finish building templates/index.html.</p>",
            status_code=404
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
