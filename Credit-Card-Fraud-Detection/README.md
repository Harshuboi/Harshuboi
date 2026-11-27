# Credit Card Fraud Detection

A reproducible machine-learning workflow for screening potentially fraudulent credit-card transactions in a highly imbalanced dataset.

## Features
- Preprocessing and scaling of transaction Amount and Time
- Stratified train/test split
- Class-imbalance handling with class weights or training-only random undersampling
- Logistic Regression and Random Forest classifiers
- Precision, recall, F1, ROC-AUC, PR-AUC (average precision), confusion matrix
- Saved model pipeline and batch prediction utility
- Streamlit interface for scoring CSV uploads

## Dataset
Download the publicly available Credit Card Fraud Detection dataset (European cardholders, September 2013) from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in this project folder. The dataset is not included because it is large and subject to the source's terms. Expected columns: `Time`, `Amount`, `V1`–`V28`, and `Class` (0 = legitimate, 1 = fraud).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
Run from this project directory:
```bash
python src/train.py --data creditcard.csv --model logistic --imbalance class_weight
```
Options: `--model logistic|random_forest`, `--imbalance class_weight|undersample`.

Training-only undersampling prevents data leakage into the test set. The model and evaluation metrics are saved under `models/` (generated locally; not committed).

## Score a CSV
After training:
```bash
python src/predict.py transactions.csv --output predictions.csv
```
The input must contain the same feature columns as the training data. The output includes fraud probability and predicted class.

## Dashboard
```bash
streamlit run app.py
```
Upload a CSV with matching feature columns. Do not upload real cardholder data or personally identifiable information.

## Evaluation
Accuracy alone can be misleading for rare fraud events. Focus on precision, recall, F1, and PR-AUC; adjust the alert threshold based on the use case. A prediction is a risk score, not proof of fraud, and should not be the sole basis for adverse decisions.
