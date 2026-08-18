"""
Module: app.py
Programmer: Tim Walewangko
Description: Streamlit interface that collects patient and screening inputs,
loads the trained HGBC pipeline, and displays a cancer prediction.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "breast_cancer_prediction.csv"
MODEL_PATH = ROOT / "models" / "breast_cancer_hgbc.joblib"
DROP_COLUMNS = ["Cancer", "Patient_ID", "Biopsy_Result", "Cancer_Stage"]

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️")
st.title("Breast Cancer Prediction")
st.caption("HGBC prediction using screening and patient information")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
X = df.drop(columns=DROP_COLUMNS)
default = X.iloc[0]

if not MODEL_PATH.exists():
    st.error("Model not found. Run breast_cancer.ipynb through the final export cell.")
    st.stop()

model = load_model()
values = {}
columns = st.columns(3)

for i, name in enumerate(X.columns):
    with columns[i % 3]:
        if pd.api.types.is_numeric_dtype(X[name]):
            if pd.api.types.is_integer_dtype(X[name]):
                values[name] = st.number_input(name.replace("_", " "), value=int(default[name]), step=1)
            else:
                values[name] = st.number_input(name.replace("_", " "), value=float(default[name]), step=0.1)
        else:
            options = X[name].dropna().unique().tolist()
            values[name] = st.selectbox(name.replace("_", " "), options, index=options.index(default[name]))

if st.button("Predict", type="primary"):
    patient = pd.DataFrame([values], columns=X.columns)
    probability = model.predict_proba(patient)[0, 1]
    prediction = "Cancer" if probability >= 0.5 else "No Cancer"

    if prediction == "Cancer":
        st.error(prediction)
    else:
        st.success(prediction)
    st.metric("Cancer probability", f"{probability:.1%}")
    st.caption("For coursework only. This is not a medical diagnosis.")
