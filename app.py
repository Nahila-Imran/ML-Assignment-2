import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TARGET = "diagnosis"
LABELS = ["Benign (0)", "Malignant (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Nahila's Cancer Detection Dashboard", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def load_features():
    return json.load(open(MODEL_DIR / "feature_names.json"))


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("🩺 Nahila's Cancer Detection Dashboard 🩺")
st.caption("Upload test data and see how five ML models perform on tumour diagnosis."
           "Models are loaded from disk - nothing is retrained on your upload.")

features = load_features()

with st.sidebar:
    st.header("Dashboard Options")
    uploaded = st.file_uploader("Choose your test file", type="csv",
                                help=f"Needs the {len(features)} feature columns plus a '{TARGET}' column.")
    model_name = st.selectbox("Pick a classifier", list(MODEL_FILES))
    show_all = st.checkbox("Show all models side by side", value=True)
    st.markdown("---")
    st.markdown("**Target:** `diagnosis` (1 = malignant, 0 = benign)")
    st.markdown("No file handy? Use `test_data.csv` from the GitHub repo.")

if uploaded is None:
    st.info("Upload the test CSV from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

X, y = df[features], df[TARGET].astype(int)
if y.nunique() < 2:
    st.error("The target column needs both classes (0 and 1) so that AUC can be computed.")
    st.stop()

st.subheader("Data preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Features", len(features))
c3.metric("Malignant share", f"{y.mean():.1%}")
st.dataframe(df.head(10), use_container_width=True)

model = load_model(model_name)
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]
scores = compute_metrics(y, pred, prob)

st.subheader(f"Results - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, scores.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=LABELS, cmap="Purples", ax=ax, colorbar=False)
    ax.set_title(model_name)
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=LABELS, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

if show_all:
    st.subheader("All models on the uploaded data")
    rows = []
    for name in MODEL_FILES:
        m = load_model(name)
        s = compute_metrics(y, m.predict(X), m.predict_proba(X)[:, 1])
        rows.append({"ML Model Name": name, **s})
    comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#ffe0b2"), use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
