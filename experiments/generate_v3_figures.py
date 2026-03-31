"""Generate all V3 figures from saved result JSONs.

Usage:
    python experiments/generate_v3_figures.py
    python experiments/generate_v3_figures.py --results-dir results/v3
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_cost_vs_coverage(sweep_data, output_dir):
    """Expected cost per 1000 inspections vs coverage target."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in sweep_data.items():
        coverages = [r["target_coverage"] for r in results]
        costs = [r["cost_per_1000"] for r in results]
        ax.plot(coverages, costs, "o-", label=label, markersize=5)
    ax.set_xlabel("Coverage target (1 - alpha)")
    ax.set_ylabel("Expected cost per 1000 inspections")
    ax.set_title("Coverage vs. inspection cost tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path = os.path.join(output_dir, "expected_cost_vs_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_abstention_by_regime(conformal_data, output_dir):
    """Bar chart of abstention rates per regime."""
    regimes = conformal_data.get("regimes", {})
    if not regimes:
        print("  Skipping abstention plot: no regime data")
        return

    labels = list(regimes.keys())
    rates = [regimes[r]["abstention_rate"] * 100 for r in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, rates, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set_ylabel("Abstention rate (%)")
    ax.set_title(f"Abstention rate by regime (alpha={conformal_data.get('alpha', 0.05)})")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "abstention_by_regime.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_class_abstention_heatmap(conformal_data, output_dir):
    """Heatmap of per-class abstention rates across regimes."""
    regimes = conformal_data.get("regimes", {})
    if not regimes:
        print("  Skipping class abstention heatmap: no regime data")
        return

    regime_names = list(regimes.keys())
    class_names = ["no_defect", "low_severity", "high_severity"]
    matrix = np.zeros((len(regime_names), 3))
    for i, name in enumerate(regime_names):
        car = regimes[name].get("class_abstention_rates", {})
        for c in range(3):
            matrix[i, c] = car.get(str(c), car.get(c, 0.0)) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names)
    for i in range(len(regime_names)):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, label="Abstention rate (%)")
    ax.set_title("Per-class abstention rates")
    fig.tight_layout()
    path = os.path.join(output_dir, "class_abstention_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate V3 figures")
    parser.add_argument("--results-dir", default="results/v3")
    parser.add_argument("--output-dir", default="docs/figures/_generated")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Generating V3 figures")
    print("=" * 60)

    # Cost sweep figure
    cost_path = os.path.join(args.results_dir, "cost_sweep_results.json")
    if os.path.exists(cost_path):
        plot_cost_vs_coverage(_load_json(cost_path), args.output_dir)
    else:
        print(f"  Skipping cost plot: {cost_path} not found")

    # Conformal figures
    conf_path = os.path.join(args.results_dir, "conformal_evaluation.json")
    if os.path.exists(conf_path):
        data = _load_json(conf_path)
        plot_abstention_by_regime(data, args.output_dir)
        plot_class_abstention_heatmap(data, args.output_dir)
    else:
        print(f"  Skipping conformal plots: {conf_path} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
