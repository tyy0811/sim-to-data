"""Calibration analysis: reliability diagrams and ECE plotting."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reliability_diagram(y_true, y_proba, n_bins=10):
    """Compute reliability diagram data.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts) as numpy arrays of length n_bins.
        NaN for empty bins.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.full(n_bins, np.nan)
    bin_accuracies = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = mask.sum()
        if mask.sum() > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = correct[mask].mean()

    return bin_confidences, bin_accuracies, bin_counts


def plot_reliability_diagram(results_list, labels, save_path):
    """Plot reliability diagrams for multiple models.

    Args:
        results_list: List of (y_true, y_proba) tuples.
        labels: List of model names.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for (y_true, y_proba), label in zip(results_list, labels):
        conf, acc, _ = reliability_diagram(y_true, y_proba)
        valid = ~np.isnan(conf)
        ax.plot(conf[valid], acc[valid], "o-", label=label)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
