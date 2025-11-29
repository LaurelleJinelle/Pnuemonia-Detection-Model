import streamlit as st
from utils import upload_training_data, retrain_model

st.title("🔄 Retrain Model")

label = st.text_input("Enter class label ("COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral")

files = st.file_uploader(
    "Upload images for this class",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if label and files:
    if st.button("Upload Data"):
        result = upload_training_data(label, files)
        st.success(result["message"])
        st.json(result)

if st.button("Start Retraining"):
    with st.spinner("Training model... this may take a while"):
        result = retrain_model()
    st.success("Retraining Complete!")
    st.json(result)

