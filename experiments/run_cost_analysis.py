"""Cost-sensitive analysis across coverage operating points.

Sweeps alpha from 0.01 to 0.50, computes expected cost at each,
and generates expected_cost_vs_coverage figure.

Usage:
    python experiments/run_cost_analysis.py
    python experiments/run_cost_analysis.py --cost-config configs/cost_matrix.yaml
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from simtodata.evaluation.conformal import ConformalClassifier
from simtodata.evaluation.cost import (
    CostMatrix,
    compute_expected_cost,
    sweep_coverage_vs_cost,
)


def _load_result(path):
    """Load a result JSON and return (softmax, labels) arrays."""
    with open(path) as f:
        data = json.load(f)
    return np.array(data["y_proba"]), np.array(data["y_true"])


def _plot_cost_vs_coverage(sweep_results_by_regime, output_path):
    """Plot expected cost per 1000 inspections vs coverage target."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in sweep_results_by_regime.items():
        coverages = [r["target_coverage"] for r in results]
        costs = [r["cost_per_1000"] for r in results]
        ax.plot(coverages, costs, "o-", label=label, markersize=5)

    ax.set_xlabel("Coverage target (1 - alpha)")
    ax.set_ylabel("Expected cost per 1000 inspections")
    ax.set_title("Coverage vs. inspection cost tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # high coverage (safe) on left

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cost analysis sweep")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cost-config", default=None,
                        help="YAML cost matrix (default: built-in NDT)")
    args = parser.parse_args()

    if args.cost_config:
        cost_matrix = CostMatrix.from_yaml(args.cost_config)
    else:
        cost_matrix = CostMatrix.default_ndt()

    regimes = {
        "B1_source": "B1_cnn1d_source_on_source.json",
        "B2_shifted": "B2_cnn1d_source_on_shifted.json",
        "B5_shifted": "B5_cnn1d_randomized_finetune_on_shifted.json",
    }

    print("Cost-sensitive analysis")
    print("=" * 60)

    all_sweeps = {}
    for label, filename in regimes.items():
        path = os.path.join(args.results_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {label}: {path} not found")
            continue

        probs, labels = _load_result(path)
        results = sweep_coverage_vs_cost(probs, labels, cost_matrix)
        all_sweeps[label] = results

        # Find optimal operating point (lowest cost)
        best = min(results, key=lambda r: r["cost_per_1000"])
        print(f"\n  {label}:")
        print(f"    Optimal alpha:   {best['alpha']:.2f}")
        print(f"    Coverage:        {best['coverage']:.3f}")
        print(f"    Abstention:      {best['abstention_rate']*100:.1f}%")
        print(f"    Cost/1000:       {best['cost_per_1000']:.1f}")

    # Save results
    v3_dir = os.path.join(args.results_dir, "v3")
    os.makedirs(v3_dir, exist_ok=True)
    out_path = os.path.join(v3_dir, "cost_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_sweeps, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # Generate figure
    fig_dir = os.path.join("docs", "figures", "_generated")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_cost_vs_coverage(
        all_sweeps,
        os.path.join(fig_dir, "expected_cost_vs_coverage.png"),
    )

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Regime':<15} {'Best alpha':>10} {'Coverage':>10} {'Cost/1k':>10}")
    print("-" * 45)
    for label, results in all_sweeps.items():
        best = min(results, key=lambda r: r["cost_per_1000"])
        print(f"{label:<15} {best['alpha']:>10.2f} "
              f"{best['coverage']:>10.3f} {best['cost_per_1000']:>10.1f}")


if __name__ == "__main__":
    main()
