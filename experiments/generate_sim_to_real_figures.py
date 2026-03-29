"""Generate figures for sim-to-real B-scan experiments."""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simtodata.data.bscan_dataset import resize_bscan
from simtodata.data.constants import BSCAN_SHAPE
from simtodata.simulator.bscan import generate_synthetic_bscan
from simtodata.simulator.regime import load_regimes_from_yaml


def plot_synthetic_vs_real_bscans(save_path, config_path="configs/simulator.yaml",
                                   real_data_dir=None):
    """2x3 grid: top row = synthetic (source no-flaw, source flaw, shifted flaw),
    bottom row = real (if available) or randomized synthetic."""
    regimes = load_regimes_from_yaml(config_path)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Top row: synthetic
    titles_top = ["Synthetic Source\n(no flaw)", "Synthetic Source\n(flaw)",
                  "Synthetic Shifted\n(flaw)"]
    regimes_top = [regimes["source"], regimes["source"], regimes["shifted"]]
    defect_flags = [False, True, True]

    for col, (regime, has_defect, title) in enumerate(
        zip(regimes_top, defect_flags, titles_top)
    ):
        defects = None if has_defect else []
        result = generate_synthetic_bscan(
            regime, rng, n_positions=64, defects=defects, defect_prob=1.0,
        )
        img = resize_bscan(result.bscan, BSCAN_SHAPE)
        axes[0, col].imshow(img.T, aspect="auto", cmap="gray", origin="lower")
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Position")
        axes[0, col].set_ylabel("Time (samples)")

    # Bottom row: real data if available, otherwise more synthetic
    if real_data_dir and os.path.isdir(real_data_dir):
        from simtodata.data.virkkunen import VirkkunenLoader
        loader = VirkkunenLoader(real_data_dir)
        real_bscans, real_labels = loader.load_all()

        # Find one no-flaw, one flaw, one flaw
        noflaw_idx = np.where(real_labels == 0)[0]
        flaw_idx = np.where(real_labels == 1)[0]

        pairs = []
        if len(noflaw_idx) > 0:
            pairs.append((noflaw_idx[0], "Real\n(no flaw)"))
        else:
            pairs.append((0, "Real (sample 0)"))
        if len(flaw_idx) > 0:
            pairs.append((flaw_idx[0], "Real\n(flaw, example 1)"))
        if len(flaw_idx) > 1:
            pairs.append((flaw_idx[1], "Real\n(flaw, example 2)"))

        for col, (idx, title) in enumerate(pairs[:3]):
            img = resize_bscan(real_bscans[idx], BSCAN_SHAPE)
            axes[1, col].imshow(img.T, aspect="auto", cmap="gray", origin="lower")
            axes[1, col].set_title(title)
            axes[1, col].set_xlabel("Position")
            axes[1, col].set_ylabel("Time (samples)")
    else:
        # Fallback: show randomized synthetic
        titles_bot = ["Randomized\n(no flaw)", "Randomized\n(flaw)", "Randomized\n(flaw)"]
        for col, (has_defect, title) in enumerate(zip([False, True, True], titles_bot)):
            defects = None if has_defect else []
            result = generate_synthetic_bscan(
                regimes["randomized"], rng, n_positions=64, defects=defects,
                defect_prob=1.0,
            )
            img = resize_bscan(result.bscan, BSCAN_SHAPE)
            axes[1, col].imshow(img.T, aspect="auto", cmap="gray", origin="lower")
            axes[1, col].set_title(title)
            axes[1, col].set_xlabel("Position")
            axes[1, col].set_ylabel("Time (samples)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_sim_to_real_results(save_path, results_dir="results/sim_to_real"):
    """Bar chart of F1 across SB1-SB3 and optionally SR1-SR2."""
    experiments = ["SB1", "SB2", "SB3", "SR1", "SR2"]
    labels_map = {
        "SB1": "Source\n-> Source",
        "SB2": "Source\n-> Shifted",
        "SB3": "Rand\n-> Shifted",
        "SR1": "Source\n-> Real",
        "SR2": "Rand\n-> Real",
    }
    file_map = {
        "SB1": "SB1_bscan_source_on_source",
        "SB2": "SB2_bscan_source_on_shifted",
        "SB3": "SB3_bscan_randomized_on_shifted",
        "SR1": "SR1_bscan_source_on_real",
        "SR2": "SR2_bscan_randomized_on_real",
    }

    names, f1s, colors = [], [], []
    color_map = {"SB1": "#4c72b0", "SB2": "#dd8452", "SB3": "#55a868",
                 "SR1": "#c44e52", "SR2": "#8172b3"}

    for exp in experiments:
        path = os.path.join(results_dir, f"{file_map[exp]}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            names.append(labels_map[exp])
            f1s.append(data["metrics"]["f1"])
            colors.append(color_map[exp])

    if not names:
        print("No results found. Run experiments first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(names)), f1s, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("F1 Score")
    ax.set_title("Sim-to-Real Transfer: B-Scan Binary Classification")
    ax.set_ylim(0, 1)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=10)

    # Add divider line between synthetic and real experiments if both present
    if len(names) > 3:
        ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
        ax.text(1.0, 0.95, "Synthetic eval", ha="center", transform=ax.get_xaxis_transform(),
                fontsize=9, color="gray")
        ax.text(3.5, 0.95, "Real eval", ha="center", transform=ax.get_xaxis_transform(),
                fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sim-to-real figures")
    parser.add_argument("--real-data-dir", default=None)
    parser.add_argument("--results-dir", default="results/sim_to_real")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_synthetic_vs_real_bscans(
        os.path.join(args.output_dir, "sim_vs_real_bscans.png"),
        real_data_dir=args.real_data_dir,
    )
    plot_sim_to_real_results(
        os.path.join(args.output_dir, "sim_to_real_results.png"),
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
