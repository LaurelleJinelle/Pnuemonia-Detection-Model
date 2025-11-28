import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

st.title("📊 Model Visualizations & Metrics")

st.write("""
This page displays visualizations from your most recent model training or retraining.  
Make sure you have trained/retrained the model before using this page.
""")

# Retrieve training history stored in session_state
history = st.session_state.get("history", None)

if history is None:
    st.warning("⚠️ No training history found. Retrain the model first.")
    st.stop()


# -------------------------
# TRAINING CURVES
# -------------------------
st.header("📈 Training & Validation Curves")

phase1 = history["phase1"]
phase2 = history["phase2"]

# Phase 1 Plot: Accuracy
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(phase1["accuracy"], label="Train Accuracy (Phase 1)")
ax.plot(phase1["val_accuracy"], label="Validation Accuracy (Phase 1)")
ax.set_title("Training vs Validation Accuracy (Phase 1)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# Phase 2 Plot: Accuracy
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(phase2["accuracy"], label="Train Accuracy (Phase 2)")
ax.plot(phase2["val_accuracy"], label="Validation Accuracy (Phase 2)")
ax.set_title("Fine-tuning Accuracy (Phase 2)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)


# LOSS CURVES
st.subheader("📉 Training vs Validation Loss")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(phase1["loss"], label="Train Loss (Phase 1)")
ax.plot(phase1["val_loss"], label="Validation Loss (Phase 1)")
ax.plot(phase2["loss"], label="Train Loss (Phase 2)")
ax.plot(phase2["val_loss"], label="Validation Loss (Phase 2)")
ax.set_title("Training vs Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)


# -------------------------
# CONFUSION MATRIX + METRICS
# -------------------------
st.header("🧪 Evaluation Metrics")

# Check if you already stored y_true and y_pred (optional)
if "y_true" in st.session_state and "y_pred" in st.session_state:

    y_true = st.session_state["y_true"]
    y_pred = st.session_state["y_pred"]
    classes = history["classes"]

    st.subheader("🔢 Classification Report")
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    st.json(report)

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.info("Prediction labels not available yet. They will appear here after you run evaluation in your notebook.")

