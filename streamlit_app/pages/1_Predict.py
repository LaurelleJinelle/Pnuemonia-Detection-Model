import streamlit as st
from utils import predict_image

st.title("🔍 X-ray Image Prediction")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Sending to model..."):
            result = predict_image(uploaded_file)

        st.subheader("Prediction Result")
        st.json(result)
