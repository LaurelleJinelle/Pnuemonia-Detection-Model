import requests

API_URL = "https://pnuemonia-detection-model.onrender.com"


def predict_image(file):
    """Send image to FastAPI /predict endpoint."""
    # Auto-detect MIME type
    mime = "image/png" if file.name.lower().endswith("png") else "image/jpeg"

    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (file.name, file, mime)}
    )

    try:
        return response.json()
    except Exception:
        return {"error": response.text}


def upload_training_data(label, files):
    """Send multiple images to the FastAPI /upload-data endpoint."""

    upload_files = []
    for f in files:
        mime = "image/png" if f.name.lower().endswith("png") else "image/jpeg"
        upload_files.append(("files", (f.name, f, mime)))

    data = {"label": label}

    response = requests.post(
        f"{API_URL}/upload-data",
        data=data,
        files=upload_files
    )

    try:
        return response.json()
    except Exception:
        return {"error": response.text}


def retrain_model():
    """Call /retrain and ensure JSON-safe output."""
    response = requests.post(f"{API_URL}/retrain")
    try:
        return response.json()
    except Exception:
        return {"error": response.text}


def get_status():
    response = requests.get(f"{API_URL}/status")
    try:
        return response.json()
    except Exception:
        return {"error": response.text}
