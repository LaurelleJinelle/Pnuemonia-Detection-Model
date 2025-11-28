import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_for_prediction
import threading

predict_lock = threading.Lock()

_model = None
_predict_fn = None

# ✔ Correct 4-class mapping
_class_names = [
    "COVID-19",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral"
]

def load_model(model_path: str):
    global _model, _predict_fn
    print(f"[INFO] Loading model from {model_path}")

    # Load SavedModel (from model.export)
    _model = tf.saved_model.load(model_path)
    _predict_fn = _model.signatures["serving_default"]


def predict_image_bytes(img_bytes: bytes):
    global _predict_fn

    img = preprocess_for_prediction(img_bytes)

    # Run inference safely
    with predict_lock:
        out = _predict_fn(img)

    # SavedModel outputs a dict → get first key
    output_key = list(out.keys())[0]
    preds = out[output_key].numpy()[0]     # shape (4,)

    # Softmax outputs for 4 classes
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = _class_names[class_idx]

    return label, confidence
