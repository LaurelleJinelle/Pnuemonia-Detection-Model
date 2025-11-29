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

# Store uploaded paths for this session
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
            # Save paths in session state
            st.session_state.uploaded_paths[label] = result["paths"]

            st.success("Upload complete!")
        else:
            st.error(f"Upload failed: {result}")

# Display uploaded paths cleanly
if st.session_state.uploaded_paths:
    st.subheader("Uploaded Data Overview")
    for class_label, paths in st.session_state.uploaded_paths.items():
        st.write(f"**{class_label}**")
        for p in paths:
            st.code(p)


# --- Retrain section ---
st.subheader("Start Model Retraining")

if st.button("Start Retraining"):
    response = retrain_model()

    # Show success ONLY if retraining truly succeeded
    if "message" in response and response["message"] == "Retraining complete.":
        st.success("Retraining Complete!")
    else:
        st.error("Retraining failed!")
        st.json(response)
