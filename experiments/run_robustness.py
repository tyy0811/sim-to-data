"""Run robustness sweep across shift intensities."""

import json
import os

import joblib
import torch

from simtodata.evaluation.robustness import run_robustness_sweep
from simtodata.models.factory import model_from_config


def main():
    os.makedirs("results", exist_ok=True)

    config_path = "configs/model_cnn1d.yaml"

    # Load models
    models = []
    names = []

    # B0c: Gradient Boosting
    if os.path.exists("models/B0b_gb_source.joblib"):
        clf = joblib.load("models/B0b_gb_source.joblib")
        models.append((clf, False))
        names.append("B0c_gb")

    # B2: Source CNN
    if os.path.exists("models/B1_cnn1d_source.pt"):
        model_b2 = model_from_config(config_path)
        model_b2.load_state_dict(torch.load("models/B1_cnn1d_source.pt", weights_only=True))
        models.append((model_b2, True))
        names.append("B2_cnn1d_source")

    # B5: Randomized + finetuned CNN
    if os.path.exists("models/B5_cnn1d_randomized_finetuned.pt"):
        model_b5 = model_from_config(config_path)
        model_b5.load_state_dict(
            torch.load("models/B5_cnn1d_randomized_finetuned.pt", weights_only=True)
        )
        models.append((model_b5, True))
        names.append("B5_cnn1d_rand_ft")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n_samples = 100 if args.quick else 1000
    print("Running robustness sweep...")
    results = run_robustness_sweep(models, names, n_samples=n_samples)

    with open("results/robustness_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/robustness_sweep.json")


if __name__ == "__main__":
    main()
