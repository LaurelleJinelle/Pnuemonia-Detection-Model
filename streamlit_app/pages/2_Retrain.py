import streamlit as st
from utils import upload_training_data, retrain_model

st.title("Retrain Model")

# --- Upload section ---
st.subheader("Upload Training Images")

label = st.text_input(
    "Enter class label (COVID-19, Normal, Pneumonia-Bacterial, Pneumonia-Viral)",
    ""
)

files = st.file_uploader(
    "Upload images for this class",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Store uploaded paths for this session only
if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = {}

if st.button("Upload Data"):
    if not label.strip():
        st.error("Please enter a label.")
    elif not files:
        st.error("Please upload at least one image.")
    else:
        result = upload_training_data(label, files)

        if "paths" in result:
            st.session_state.uploaded_paths[label] = result["paths"]
            st.success("Upload complete!")
        else:
            st.error(f"Upload failed")
            st.json(result)

# Display uploaded data
if st.session_state.uploaded_paths:
    st.subheader("Uploaded Data Overview")

    for class_label, paths in st.session_state.uploaded_paths.items():
        st.write(f"### {class_label}")
        for p in paths:
            st.code(p)

# --- Retrain section ---
st.subheader("Start Model Retraining")

if st.button("Start Retraining"):
    response = retrain_model()

    # Case 1: retraining succeeded normally
    if "message" in response and response["message"] == "Retraining complete.":
        st.success("🎉 Retraining complete! Your model has been updated.")
        st.json(response)

    # Case 2: backend returned a structured error JSON
    elif "error" in response and response["error"]:
        st.error(f"❌ Retraining failed: {response['error']}")
        st.json(response)

    # Case 3: server restarted during training → HTML returned
    else:
        st.warning(
            "⚠️ The server restarted during retraining. "
            "The process likely finished, but the response was incomplete."
        )
        st.write("Raw response:")
        st.code(str(response))
