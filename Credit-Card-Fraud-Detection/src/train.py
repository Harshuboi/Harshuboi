"""Train and evaluate a credit-card fraud classification pipeline."""
from pathlib import Path
import argparse, json
import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report)

def build_pipeline(model_name, imbalance):
    preprocessor = ColumnTransformer(
        [("scale_amount_time", StandardScaler(), ["Amount", "Time"])],
        remainder="passthrough", verbose_feature_names_out=False)
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=1500, solver="liblinear",
            class_weight="balanced" if imbalance == "class_weight" else None, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2,
            class_weight="balanced" if imbalance == "class_weight" else None,
            n_jobs=-1, random_state=42)
    steps = [("preprocessor", preprocessor)]
    if imbalance == "undersample":
        steps.append(("undersampler", RandomUnderSampler(random_state=42)))
    steps.append(("classifier", clf))
    return ImbPipeline(steps)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--imbalance", choices=["class_weight", "undersample"], default="class_weight")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download creditcard.csv and place it in the project folder.")
    df = pd.read_csv(path).dropna(subset=["Class"])
    if "Class" not in df:
        raise ValueError("Dataset must include the target column 'Class'.")
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    if y.nunique() != 2 or not {"Amount", "Time"}.issubset(X.columns):
        raise ValueError("Expected binary Class target and Amount/Time features.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42)
    pipeline = build_pipeline(args.model, args.imbalance)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    score = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "model": args.model, "imbalance_strategy": args.imbalance,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "pr_auc_average_precision": float(average_precision_score(y_test, score)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)}
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": list(X.columns)}, "models/fraud_pipeline.joblib")
    Path("models/evaluation.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    print(classification_report(y_test, pred, zero_division=0))
    print("Saved model to models/fraud_pipeline.joblib")

if __name__ == "__main__":
    main()
