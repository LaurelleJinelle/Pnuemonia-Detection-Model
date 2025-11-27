import streamlit as st
from utils import get_status

st.title("📡 System Status")

status = get_status()
st.json(status)
