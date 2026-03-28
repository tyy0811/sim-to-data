"""Aggregate multi-seed results into mean +/- std."""

import argparse
import json
import os

import numpy as np

SEEDS = [42, 123, 456, 789, 1024]
EXPERIMENTS = {
    "B1": "B1_cnn1d_source_on_source",
    "B2": "B2_cnn1d_source_on_shifted",
    "B3": "B3_cnn1d_randomized_on_shifted",
    "B4": "B4_cnn1d_source_finetune_on_shifted",
    "B5": "B5_cnn1d_randomized_finetune_on_shifted",
}
METRICS = ["macro_f1", "auroc", "ece"]


def aggregate(base_dir="results/multiseed"):
    results = {}
    for short_name, file_name in EXPERIMENTS.items():
        values = {m: [] for m in METRICS}
        for seed in SEEDS:
            path = os.path.join(base_dir, f"seed_{seed}", f"{file_name}.json")
            with open(path) as f:
                data = json.load(f)
            for m in METRICS:
                values[m].append(data["metrics"][m])
        results[short_name] = {
            m: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "values": v,
            }
            for m, v in values.items()
        }

    out_path = os.path.join(base_dir, "aggregated.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    print(f"\n{'Exp':<6} {'F1':>14} {'AUROC':>14} {'ECE':>14}")
    print("-" * 50)
    for short_name in EXPERIMENTS:
        r = results[short_name]
        print(
            f"{short_name:<6} "
            f"{r['macro_f1']['mean']:.3f}\u00b1{r['macro_f1']['std']:.3f}  "
            f"{r['auroc']['mean']:.3f}\u00b1{r['auroc']['std']:.3f}  "
            f"{r['ece']['mean']:.3f}\u00b1{r['ece']['std']:.3f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--input-dir", default="results/multiseed",
                        help="Base directory with seed_* subdirectories")
    args = parser.parse_args()
    aggregate(args.input_dir)


if __name__ == "__main__":
    main()
