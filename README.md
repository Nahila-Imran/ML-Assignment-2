# Breast Tumour Classification - Five-Model Comparison

## a. Problem statement
Early and accurate classification of breast masses as benign or malignant is central to timely treatment. This project builds and compares five supervised classifiers that predict whether a tumour is malignant (1) or benign (0) from measurements of cell nuclei taken from a fine needle aspirate (FNA) image. Because missing a malignant case is costlier than a false alarm, recall and MCC are tracked alongside accuracy.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, UCI Machine Learning Repository (also distributed with scikit-learn as load_breast_cancer).
- **Instances:** 569 (357 benign, 212 malignant). Minimum requirement of 500 is met.
- **Features:** 30 numeric features (minimum requirement of 12 is met): the mean, standard error and worst value of 10 nucleus measurements - radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.
- **Target:** diagnosis: 1 = malignant, 0 = benign (re-coded so the clinically important class is the positive class). Binary classification.
- **Data quality:** No missing values.
- **Split:** Stratified 75/25 train/test split with random_state = 2026: 426 training rows and 143 test rows. The 143-row test set is exported as test_data.csv and is what the Streamlit app evaluates.
- **Preprocessing:** StandardScaler inside the scikit-learn Pipeline for Logistic Regression, KNN and SVM (fitted on training data only). Decision Tree and Gaussian Naive Bayes use raw features.

## c. GitHub repository link
https://github.com/Nahila-Imran/ML-Assignment-2

Live Streamlit app: https://ml-assignment-2-mlwevmpce4vs699txvzcqo.streamlit.app/



## d. Models used
Comparison table (evaluated on the 143-row held-out test set):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9650 | 0.9843 | 0.9800 | 0.9245 | 0.9515 | 0.9251 |
| Decision Tree | 0.9231 | 0.9401 | 0.9038 | 0.8868 | 0.8952 | 0.8346 |
| KNN | 0.9580 | 0.9802 | 0.9608 | 0.9245 | 0.9423 | 0.9098 |
| Naive Bayes | 0.9301 | 0.9881 | 0.9216 | 0.8868 | 0.9038 | 0.8493 |
| Support Vector Machine | 0.9720 | 0.9881 | 0.9623 | 0.9623 | 0.9623 | 0.9400 |

Settings: **Logistic Regression** - max_iter = 1000, with feature scaling. **Decision Tree** - max_depth = 5, min_samples_leaf = 3. **KNN** - k = 7, with feature scaling. **Naive Bayes** - GaussianNB, default settings. **Support Vector Machine** - RBF kernel, C = 2.0, probability = True (needed for AUC), with feature scaling.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed very well. It is fast, simple and easy to understand, but it missed 4 cancer cases. |
| Decision Tree | Decision Tree was the weakest model in this comparison. It made more mistakes than the other models. |
| KNN | KNN did well because after scaling, patients with similar measurements sit close together. It made 2 false alarms but missed 4 cancers, which is a concern in a medical setting. |
| Naive Bayes | ANaive Bayes can rank cancer and non-cancer cases very well, but its final predictions were not as accurate because its default decision threshold was not suitable. |
| Support Vector Machine | SVM performed the best overall. It correctly identified most cancer and non-cancer cases and made very few mistakes. |
| Overall winner | After comparing five machine learning models, Support Vector Machine (SVM) achieved the best overall performance, with 97.2% accuracy, 96.2% recall and a 0.962 F1-score. It also made only 2 false positive and 2 false negative predictions.|

Logistic Regression and KNN also performed well, while Decision Tree showed the weakest performance.
Therefore, SVM is the best-performing model in this experiment. However, further testing with a larger dataset and proper parameter tuning is needed before considering it for real medical use.

## How to run
```
pip install -r requirements.txt
python train_models.py      # retrains and re-exports models + test_data.csv
streamlit run app.py
```
Upload `test_data.csv` in the app, pick a model, and view the metrics, confusion matrix and classification report.
