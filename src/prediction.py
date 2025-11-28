import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_for_prediction
import threading

predict_lock = threading.Lock()
_model = None
_predict_fn = None

_class_names = ["NORMAL", "PNEUMONIA"]

def load_model(model_path: str):
    global _model, _predict_fn
    print(f"[INFO] Loading model from {model_path}")
    
    # Load SavedModel exported via model.export()
    _model = tf.saved_model.load(model_path)

    # Get prediction function
    _predict_fn = _model.signatures["serving_default"]


def predict_image_bytes(img_bytes: bytes):
    global _predict_fn

    # Preprocess image: outputs Tensor of shape (1,224,224,3)
    img = preprocess_for_prediction(img_bytes)

    # Run prediction safely
    with predict_lock:
        out = _predict_fn(img)

    # "probs" is usually the name of output (inspect keys)
    output_key = list(out.keys())[0]
    preds = out[output_key].numpy()[0]

    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = _class_names[class_idx]

    return label, confidence
