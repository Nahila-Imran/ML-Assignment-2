# Breast Tumour Classification - Five-Model Comparison

## a. Problem statement

Doctors need to know early whether a breast lump is harmless (benign) or cancerous (malignant). In this project, I train five machine learning models to predict this from measurements of cells. The cells are taken from a small needle sample of the lump.

Missing a cancer case is more serious than a false alarm. So I check recall and MCC along with accuracy.

## b. Dataset description

| Item | Details |
|---|---|
| Source | Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository. It is also available in scikit-learn as `load_breast_cancer`. |
| Rows | 569 patients (357 benign, 212 malignant). This meets the minimum of 500. |
| Features | 30 number columns. This meets the minimum of 12. They describe the cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each has a mean, a standard error and a worst value. |
| Target | `diagnosis`: 1 = malignant, 0 = benign. I made malignant the "positive" class because it is the important one to catch. This is a two-class (binary) problem. |
| Missing values | None. |
| Train/test split | 75% for training (426 rows) and 25% for testing (143 rows). The split keeps the same benign/malignant ratio in both parts. The test rows are saved as `test_data.csv`, and the Streamlit app uses this file. |
| Preprocessing | Logistic Regression, KNN and SVM use feature scaling (`StandardScaler`), learned from the training data only. Decision Tree and Naive Bayes use the raw values. |

## c. GitHub repository link

https://github.com/Nahila-Imran/ML-Assignment-2

Live Streamlit app: https://ml-assignment-2-mlwevmpce4vs699txvzcqo.streamlit.app/

```
project-folder/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|   |-- feature_names.json
|   |-- metrics_summary.csv
```

## d. Models used

All five models were tested on the same 143 test rows.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9650 | 0.9843 | 0.9800 | 0.9245 | 0.9515 | 0.9251 |
| Decision Tree | 0.9231 | 0.9401 | 0.9038 | 0.8868 | 0.8952 | 0.8346 |
| KNN | 0.9580 | 0.9802 | 0.9608 | 0.9245 | 0.9423 | 0.9098 |
| Naive Bayes | 0.9301 | 0.9881 | 0.9216 | 0.8868 | 0.9038 | 0.8493 |
| Support Vector Machine | 0.9720 | 0.9881 | 0.9623 | 0.9623 | 0.9623 | 0.9400 |

**Model settings**

- Logistic Regression: maximum 1000 iterations, with scaling.
- Decision Tree: maximum depth 5, at least 3 samples in each leaf.
- KNN: 7 neighbours, with scaling.
- Naive Bayes: Gaussian type, default settings.
- Support Vector Machine: RBF kernel, C = 2.0, probability turned on (needed for AUC), with scaling.

### Observations on model performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Very good: 96.5% accuracy and the highest precision (0.980). It raised only 1 false alarm. It missed 4 cancer cases. This shows the two classes can be separated well by a straight line once the data is scaled. It is also fast and easy to explain. |
| Decision Tree | The weakest model on every metric (accuracy 92.3%, AUC 0.940). It raised 5 false alarms and missed 6 cancer cases. One small tree can change a lot with different data, and it gives rough probability scores. That is why its AUC is lower. |
| KNN | Good: 95.8% accuracy and 0.980 AUC. It raised 2 false alarms and missed 4 cancer cases. It works well only after scaling. It is also slower at prediction because it keeps all the training data. |
| Naive Bayes | Accuracy is average (93.0%), but its AUC (0.988) is among the best. This means it ranks risky cases well, but its cut-off point is not ideal. It missed 6 cancer cases. It assumes features are independent, but features like radius, perimeter and area are closely related. |
| Support Vector Machine | The best on accuracy (97.2%), recall (0.962), F1 (0.962) and MCC (0.940). Its AUC (0.988) is tied with Naive Bayes. It made only 2 false alarms and missed only 2 cancer cases. The RBF kernel helps it follow a slightly curved boundary between the classes. |
| Overall winner | **Support Vector Machine.** It is best on four metrics and missed the fewest cancer cases. **Please note:** the test set is small (143 rows), and one case changes accuracy by about 0.7%. SVM, Logistic Regression and KNN are very close, so the gap between them may not be real. I also set the model settings by hand and did not tune them with cross-validation. |

## How to run

```
pip install -r requirements.txt
python train_models.py      # retrains the models and re-creates test_data.csv
streamlit run app.py
```

Upload `test_data.csv` in the app, pick a model, and view the metrics, confusion matrix and classification report.
