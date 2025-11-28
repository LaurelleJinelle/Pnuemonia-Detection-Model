import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_for_prediction
from keras import models as keras_models 
import threading
predict_lock = threading.Lock()

_model = None
_class_names = ["NORMAL", "PNEUMONIA"]   

def load_model(model_path: str):
    global _model
    print(f"[INFO] Loading model from {model_path}")
    _model = keras_models.load_model(model_path)

def predict_image_bytes(img_bytes: bytes):
    global _model
    
    img = preprocess_for_prediction(img_bytes)
    with predict_lock:
        preds = _model.predict(img)[0]

    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = _class_names[class_idx]

    return label, confidence




