"""Run adaptation efficiency experiment."""

import json
import os

import torch
from torch.utils.data import DataLoader

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.adaptation_curve import run_adaptation_sweep
from simtodata.models.factory import model_from_config


def main():
    os.makedirs("results", exist_ok=True)
    config_path = "configs/model_cnn1d.yaml"
    norm = Normalize()

    adapt_data = InspectionDataset("data/shifted_adapt.npz", transform=norm)
    shifted_test = InspectionDataset("data/shifted_test.npz", transform=norm)
    eval_loader = DataLoader(shifted_test, batch_size=256)

    model_template = model_from_config(config_path)
    all_results = {}

    # Source-pretrained (B4-style)
    if os.path.exists("models/B1_cnn1d_source.pt"):
        print("Adaptation curve: source-pretrained...")
        results = run_adaptation_sweep(
            model_template, "models/B1_cnn1d_source.pt", adapt_data, eval_loader
        )
        all_results["source_pretrained"] = results

    # Randomized-pretrained (B5-style)
    if os.path.exists("models/B3_cnn1d_randomized.pt"):
        print("Adaptation curve: randomized-pretrained...")
        results = run_adaptation_sweep(
            model_template, "models/B3_cnn1d_randomized.pt", adapt_data, eval_loader
        )
        all_results["randomized_pretrained"] = results

    with open("results/adaptation_curve.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved: results/adaptation_curve.json")


if __name__ == "__main__":
    main()
