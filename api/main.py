import os
import time
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from src import prediction
from src import model as model_utils   # your retraining utilities

# --------------------
# Directory structure
# --------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RETRAIN_DIR = os.path.join(BASE_DIR, "retrain_data")

os.makedirs(RETRAIN_DIR, exist_ok=True)

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Global API State
# --------------------
start_time = time.time()
num_predictions = 0
last_retrain_time = None

# ✔ Load fine-tuned SavedModel
current_model_path = "/app/models/mobilenet_pneumonia_finetuned_model"
prediction.load_model(current_model_path)

# --------------------
# ROUTES
# --------------------

@app.get("/status")
def get_status():
    """Return API health, model uptime, retrain time, prediction count."""
    return {
        "uptime_seconds": round(time.time() - start_time, 2),
        "num_predictions": num_predictions,
        "last_retrain_time": last_retrain_time,
        "current_model_path": current_model_path,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Predict pneumonia class for a single uploaded image."""

    global num_predictions

    img_bytes = await file.read()
    label, confidence = prediction.predict_image_bytes(img_bytes)

    num_predictions += 1

    return {
        "filename": file.filename,
        "label": label,
        "confidence": round(confidence, 4)
    }


@app.post("/upload-data")
async def upload_data(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Upload new training images for retraining."""

    label_dir = os.path.join(RETRAIN_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    saved_files = []
    for f in files:
        content = await f.read()
        filename = f.filename.replace(" ", "_")
        path = os.path.join(label_dir, filename)
        with open(path, "wb") as out:
            out.write(content)
        saved_files.append(path)

    return {
        "message": f"Saved {len(saved_files)} images under label '{label}'.",
        "paths": saved_files
    }


@app.post("/retrain")
async def retrain():
    """Trigger full retraining of the model using uploaded data."""

    global last_retrain_time, current_model_path

    # This function must return a SavedModel directory path
    new_model_path, history = model_utils.retrain_model(RETRAIN_DIR)

    # Update global model reference
    prediction.load_model(new_model_path)
    current_model_path = new_model_path

    last_retrain_time = datetime.utcnow().isoformat() + "Z"

    return {
        "message": "Retraining complete.",
        "new_model_path": new_model_path,
        "history": history,

    }





