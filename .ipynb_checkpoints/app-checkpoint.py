import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_fraud_pipeline.pkl")   

model = load_model()

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a dataset and check fraud predictions with performance metrics.")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Uploaded Data")
    st.write(df.head())

    # -----------------------------
    # Predictions
    # -----------------------------
    X = df.drop("Class", axis=1, errors="ignore")  # Features
    if "Class" in df.columns:
        y_true = df["Class"]
    else:
        y_true = None

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    df["Prediction"] = y_pred
    st.subheader("🔍 Predictions")
    st.write(df.head())

    # -----------------------------
    # Metrics
    # -----------------------------
    if y_true is not None:
        st.subheader("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.write("ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred))

else:
    st.info("👆 Upload a CSV file to start.")
