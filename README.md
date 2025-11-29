** Pneumonia Detection Model — End-to-End MLOps Pipeline **


📌 YouTube Link: https://youtu.be/NkDrN6hW0Jo

🌐 Deployed URLs 
🚀 FastAPI Backend (Render): https://pnuemonia-detection-model.onrender.com 
🎨 Streamlit UI: https://pnuemonia-detection-model.streamlit.app/

** Project Description **

This project implements a full Machine Learning Pipeline & MLOps workflow for detecting pneumonia from chest X-ray images.

The system includes:
✔ Image dataset preprocessing
✔ Transfer learning using MobileNetV2
✔ Model evaluation (accuracy, precision, recall, F1-score)
✔ Cloud-deployed inference API (FastAPI + Render)
✔ Cloud-deployed frontend (Streamlit)
✔ Retraining pipeline (user uploads → retrain → updated model)
✔ Dataset visualizations & model insights
✔ Load Testing using Locust
  ✔ Monitoring: uptime, prediction count, active model version

This showcases a production-ready MLOps workflow.

** How to Set Up the Project (Local Development) **

Follow these steps if running locally.

1️⃣ Clone the Repository
```
git clone https://github.com/LaurelleJinelle/Pnuemonia-Detection-Model.git
cd YOUR_REPO_NAME
```

2️⃣ Install API Dependencies (TensorFlow, FastAPI)
`pip install -r requirements_api.txt`

3️⃣ Start the FastAPI Backend
`uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
Then open in browser:
http://localhost:8000/docs

4️⃣ Install Streamlit UI Dependencies
`pip install -r streamlit_app/requirements.txt`

5️⃣ Start the Streamlit UI
`streamlit run streamlit_app/app.py`
The UI will open in your browser.

📡 API Endpoints Overview
Endpoint	Method	Description
/predict - POST	Upload an image → get prediction
/upload-data - POST	Upload new images for retraining
/retrain - POST Trigger model retraining
/status - GET	API uptime, model version, prediction count

The backend loads: models/mobilenet_pneumonia_finetuned.h5

** Model Development (Training & Evaluation) **

Two experiments were conducted:

✔ Experiment 1: Frozen MobileNetV2
Trained only the classification head

Metrics achieved:
Accuracy: ~0.74
Precision, Recall, F1: computed in notebook

✔ Experiment 2: Fine-Tuning MobileNetV2
Unfroze top MobileNetV2 layers, and Fine-tuned with low learning rate

Improved metrics:
Accuracy: ~0.81
Higher precision, recall, and F1-score computed in notebook

Saved as:

models/mobilenet_pneumonia_finetuned.h5

All evaluation steps (classification report, confusion matrix) are inside the notebook.

📊 Dataset Visualizations (Notebook)

The following visualizations were implemented:

✔ Class distribution
✔ Sample images per class
✔ Pixel intensity histogram (brightness)
✔ Edge density (texture feature)
✔ Training curves

Interpretations are included in the notebook.

🔁 Retraining Pipeline (MLOps)

Users can:
Upload new labeled training images
Call /upload-data
Call /retrain - Backend retrains MobileNetV2
Saves a new .h5 model
API hot-reloads the new model automatically
Retraining data is stored in: retrain_data/

🚦 System Status Monitoring

The Streamlit UI shows:

API uptime
Number of predictions served
Last retraining time
Current model path

🧪 Flood Request Simulation (Locust Load Testing)
A full flood test was performed using Locust against the deployed FastAPI API.
The following script was used:
`locust -f locustfile.py --host=https://pnuemonia-detection-model.onrender.com`

<img width="1824" height="557" alt="Screenshot 2025-11-29 111015" src="https://github.com/user-attachments/assets/a1b1a8a3-5a4c-493b-bd8a-a0167d487a13" />
