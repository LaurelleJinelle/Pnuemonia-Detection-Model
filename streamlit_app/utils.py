import requests

API_URL = "https://pnuemonia-detection-model.onrender.com"


def predict_image(file):
    """Send image to FastAPI /predict endpoint."""
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (file.name, file, "image/jpeg")}
    )
    return response.json()


def upload_training_data(label, files):
    """Send files to FastAPI /upload-data."""
    upload_files = [("files", (f.name, f, "image/jpeg")) for f in files]
    data = {"label": label}
    response = requests.post(f"{API_URL}/upload-data", data=data, files=upload_files)
    return response.json()


def retrain_model():
    """Trigger the /retrain endpoint."""
    response = requests.post(f"{API_URL}/retrain")
    return response.json()


def get_status():
    """Get uptime, predictions count, current model."""
    response = requests.get(f"{API_URL}/status")
    return response.json()

