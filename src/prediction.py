import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_for_prediction

_model = None
_class_names = ["NORMAL", "PNEUMONIA"]   # adjust based on your dataset

def load_model(model_path: str):
    global _model
    print(f"[INFO] Loading model from {model_path}")
    _model = tf.keras.models.load_model(model_path)

def predict_image_bytes(img_bytes: bytes):
    img = preprocess_for_prediction(img_bytes)
    preds = _model.predict(img)[0]
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = _class_names[class_idx]
    return label, confidence
