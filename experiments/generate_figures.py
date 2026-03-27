"""Generate all figures from result JSONs."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simtodata.data.generate import generate_dataset
from simtodata.simulator.regime import RegimeConfig


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_robustness_curve(results_path, save_path):
    """Plot F1 vs shift intensity for multiple models."""
    results = _load_json(results_path)
    intensities = list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(results[intensities[0]].keys())

    for name in model_names:
        f1s = [results[intensity][name]["macro_f1"] for intensity in intensities]
        ax.plot(range(len(intensities)), f1s, "o-", label=name)

    ax.set_xticks(range(len(intensities)))
    ax.set_xticklabels(intensities)
    ax.set_xlabel("Shift Intensity")
    ax.set_ylabel("Macro F1")
    ax.set_title("Robustness Under Increasing Domain Shift")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_adaptation_curve(results_path, save_path):
    """Plot F1 vs fine-tune sample count."""
    results = _load_json(results_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, data in results.items():
        counts = sorted([int(k) for k in data.keys()])
        means = [data[str(c)]["mean_f1"] for c in counts]
        stds = [data[str(c)]["std_f1"] for c in counts]
        ax.errorbar(counts, means, yerr=stds, fmt="o-", label=strategy, capsize=3)

    ax.set_xlabel("Fine-tune Samples")
    ax.set_ylabel("Macro F1 on Shifted Test")
    ax.set_title("Adaptation Efficiency")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_example_traces(save_path):
    """Plot example traces: 3 classes x 2 regimes."""
    source = RegimeConfig("source", (15, 25), (5800, 6200), (0.01, 0.05), (2, 5),
                          (0.5, 1.5), (2, 28), (0.1, 0.8), (25, 35), (0, 0), (1, 1),
                          (0, 0), (0, 0), (0, 0))
    shifted = RegimeConfig("shifted", (15, 25), (5500, 6500), (0.01, 0.10), (1.5, 7),
                           (0.3, 2), (2, 28), (0.05, 0.9), (8, 15), (0.1, 0.2), (0.8, 1.2),
                           (0, 2), (1, 2), (5, 15))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    class_names = ["No Defect", "Low Severity", "High Severity"]

    for row, (regime, regime_name) in enumerate([(source, "Source"), (shifted, "Shifted")]):
        for col, cls_name in enumerate(class_names):
            dists = [
                {"no_defect": 1.0, "low_severity": 0.0, "high_severity": 0.0},
                {"no_defect": 0.0, "low_severity": 1.0, "high_severity": 0.0},
                {"no_defect": 0.0, "low_severity": 0.0, "high_severity": 1.0},
            ]
            data = generate_dataset(regime, 1, seed=42 + col, class_distribution=dists[col])
            axes[row, col].plot(data["signals"][0], linewidth=0.5)
            axes[row, col].set_title(f"{regime_name} - {cls_name}")
            axes[row, col].set_xlabel("Sample")
            axes[row, col].set_ylabel("Amplitude")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results_dir, save_path):
    """Plot confusion matrices for B1, B2, B5 (requires saved y_true/y_pred)."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    exp_names = ["B1_cnn1d_source_on_source", "B2_cnn1d_source_on_shifted",
                 "B5_cnn1d_randomized_finetune_on_shifted"]
    titles = ["B1: Source→Source", "B2: Source→Shifted", "B5: Rand+FT→Shifted"]
    class_labels = ["No Defect", "Low", "High"]

    for ax, exp_name, title in zip(axes, exp_names, titles):
        path = os.path.join(results_dir, f"{exp_name}.json")
        if not os.path.exists(path):
            ax.set_title(f"{title}\n(not available)")
            continue
        result = _load_json(path)
        if "y_true" in result and "y_pred" in result:
            cm = confusion_matrix(result["y_true"], result["y_pred"], normalize="true")
            disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
            disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
            ax.set_title(title)
        else:
            # Fallback: plot per-class recall bars
            per_class = result["metrics"]["per_class"]
            recalls = per_class["recall"]
            ax.bar(class_labels, recalls)
            ax.set_ylabel("Recall")
            ax.set_title(title)
            ax.set_ylim(0, 1)
            for i, v in enumerate(recalls):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs("docs/figures", exist_ok=True)

    if os.path.exists("results/robustness_sweep.json"):
        print("Plotting robustness curve...")
        plot_robustness_curve("results/robustness_sweep.json", "docs/figures/robustness_curve.png")

    if os.path.exists("results/adaptation_curve.json"):
        print("Plotting adaptation curve...")
        plot_adaptation_curve("results/adaptation_curve.json", "docs/figures/adaptation_curve.png")

    print("Plotting example traces...")
    plot_example_traces("docs/figures/example_traces.png")

    print("Plotting confusion matrices...")
    plot_confusion_matrices("results", "docs/figures/confusion_matrices.png")

    print("All figures generated.")


if __name__ == "__main__":
    main()
