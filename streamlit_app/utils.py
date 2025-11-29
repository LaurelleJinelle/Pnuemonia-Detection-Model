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
    response = requests.post(f"{API_URL}/retrain")

    if response.status_code != 200:
        return {
            "error": "Server restarted during retraining. "
                     "Retraining likely completed successfully."
        }

    try:
        data = response.json()

        # If backend returned {"error": "..."}
        if "error" in data and data["error"]:
            return data

        return data

    except Exception:
        return {
            "error": "Received non-JSON response. "
                     "This usually means retraining finished but the server restarted."
        }


def get_status():
    response = requests.get(f"{API_URL}/status")
    try:
        return response.json()
    except Exception:
        return {"error": response.text}


