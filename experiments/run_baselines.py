"""Run baseline experiments B0a-B0c."""

import json
import os

import joblib
import numpy as np

from simtodata.evaluation.metrics import compute_all_metrics
from simtodata.features.extract import extract_features_batch
from simtodata.models.baselines import create_baseline


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    source_train = np.load("data/source_train.npz")
    source_test = np.load("data/source_test.npz")
    shifted_test = np.load("data/shifted_test.npz")

    print("Extracting features...")
    train_feat = extract_features_batch(source_train["signals"])
    source_test_feat = extract_features_batch(source_test["signals"])
    shifted_test_feat = extract_features_batch(shifted_test["signals"])

    # B0a: LogReg on source
    print("B0a: Logistic Regression...")
    clf_lr = create_baseline("logistic_regression")
    clf_lr.fit(train_feat, source_train["labels"])
    preds = clf_lr.predict(source_test_feat)
    probs = clf_lr.predict_proba(source_test_feat)
    metrics = compute_all_metrics(source_test["labels"], preds, probs)
    print(f"  F1: {metrics['macro_f1']:.4f}")
    with open("results/B0a_logreg_source_on_source.json", "w") as f:
        json.dump({"name": "B0a", "metrics": metrics}, f, indent=2)

    # B0b: GradBoost on source
    print("B0b: Gradient Boosting...")
    clf_gb = create_baseline("gradient_boosting")
    clf_gb.fit(train_feat, source_train["labels"])
    joblib.dump(clf_gb, "models/B0b_gb_source.joblib")
    preds = clf_gb.predict(source_test_feat)
    probs = clf_gb.predict_proba(source_test_feat)
    metrics = compute_all_metrics(source_test["labels"], preds, probs)
    print(f"  F1: {metrics['macro_f1']:.4f}")
    with open("results/B0b_gb_source_on_source.json", "w") as f:
        json.dump({"name": "B0b", "metrics": metrics}, f, indent=2)

    # B0c: GradBoost on shifted
    print("B0c: GradBoost -> shifted...")
    preds = clf_gb.predict(shifted_test_feat)
    probs = clf_gb.predict_proba(shifted_test_feat)
    metrics = compute_all_metrics(shifted_test["labels"], preds, probs)
    print(f"  F1: {metrics['macro_f1']:.4f}")
    with open("results/B0c_gb_source_on_shifted.json", "w") as f:
        json.dump({"name": "B0c", "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
