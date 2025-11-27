import streamlit as st

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Pneumonia Detection System")
st.write("Welcome! Use the sidebar to navigate between pages:")
st.write("""
- **Predict**: Upload an X-ray image and get model prediction  
- **Retrain**: Upload new data and retrain your model  
- **Visualizations**: View model insights and evaluation  
- **Status**: View model uptime and API stats  
""")
