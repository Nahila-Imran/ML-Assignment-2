"""Train five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
save them, export a held-out test CSV and a metrics summary."""
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 2026
TARGET = "diagnosis"          # 1 = malignant, 0 = benign (re-coded below)

raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# sklearn encodes malignant=0, benign=1; flip so that malignant (the class we
# most want to catch) is the positive class = 1
df[TARGET] = 1 - df["target"]
df = df.drop(columns="target")

X, y = df.drop(columns=TARGET), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, probability=True, random_state=SEED)),
}

def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")
    rows.append({"ML Model Name": name, **evaluate(model, X_test, y_test)})

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("model/metrics_summary.csv", index=False)
json.dump(list(X.columns), open("model/feature_names.json", "w"))

test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv("test_data.csv", index=False)

print(f"Total rows: {len(df)}, features: {X.shape[1]}, train: {len(X_train)}, test: {len(X_test)}")
print(y.value_counts().rename({0: 'benign', 1: 'malignant'}).to_string())
print(metrics.to_string(index=False))
