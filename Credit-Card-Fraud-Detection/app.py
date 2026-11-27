"""Streamlit dashboard for transaction risk scoring."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

MODEL = Path("models/fraud_pipeline.joblib")
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")
st.title("Credit Card Fraud Detection")
st.caption("Transaction risk screening; a model score is not a definitive fraud determination.")
if not MODEL.exists():
    st.warning("Model not found. Train first: python src/train.py --data creditcard.csv")
    st.stop()
artifact = joblib.load(MODEL)
pipeline, features = artifact["pipeline"], artifact["feature_columns"]
uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05)
if uploaded:
    data = pd.read_csv(uploaded)
    missing = sorted(set(features) - set(data.columns))
    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()
    scores = pipeline.predict_proba(data[features])[:, 1]
    pred = (scores >= threshold).astype(int)
    result = data.copy()
    result["fraud_probability"] = scores
    result["predicted_class"] = pred
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(result):,}")
    c2.metric("Flagged", f"{int(pred.sum()):,}")
    c3.metric("Flagged rate", f"{pred.mean():.2%}" if len(pred) else "0%")
    st.dataframe(result, use_container_width=True)
    st.download_button("Download scored CSV", result.to_csv(index=False), "fraud_predictions.csv", "text/csv")
    if "Class" in data:
        st.subheader("Evaluation on uploaded labeled data")
        st.dataframe(pd.DataFrame(classification_report(data["Class"].astype(int), pred,
            output_dict=True, zero_division=0)).transpose(), use_container_width=True)
        st.write("Confusion matrix")
        st.write(confusion_matrix(data["Class"].astype(int), pred))
else:
    st.info("Upload a CSV containing the model's input feature columns.")
