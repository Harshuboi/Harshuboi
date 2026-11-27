"""Batch-score transaction rows using the trained pipeline."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

MODEL_PATH = Path("models/fraud_pipeline.joblib")

def score_transactions(input_csv, output_csv="predictions.csv", threshold=0.5):
    artifact = joblib.load(MODEL_PATH)
    model, features = artifact["pipeline"], artifact["feature_columns"]
    data = pd.read_csv(input_csv)
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    probabilities = model.predict_proba(data[features])[:, 1]
    result = data.copy()
    result["fraud_probability"] = probabilities
    result["predicted_class"] = (probabilities >= threshold).astype(int)
    result.to_csv(output_csv, index=False)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = score_transactions(args.input_csv, args.output, args.threshold)
    print(result[["fraud_probability", "predicted_class"]].head())
    print(f"Saved {args.output}")
