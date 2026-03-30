"""Cost-sensitive evaluation for safety-critical defect classification.

Computes expected inspection cost under a configurable asymmetric cost
matrix, enabling operating-point analysis: at what coverage/abstention
tradeoff is the total inspection cost minimized?

The cost values are illustrative. The framework is the contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml


@dataclass
class CostMatrix:
    """Asymmetric cost matrix for defect classification.

    Attributes:
        costs: dict mapping (true_class, predicted_class) to cost.
        review_cost: cost per abstained sample (human review).
        class_names: list of class name strings for display.
    """
    costs: dict
    review_cost: float
    class_names: list

    @classmethod
    def default_ndt(cls) -> CostMatrix:
        """Default cost matrix for 3-class NDT defect classification.

        Relative units: missed high-severity = 500, human review = 5,
        false alarm = 1.
        """
        costs = {
            (0, 0): 0,    (0, 1): 1,    (0, 2): 1,
            (1, 0): 50,   (1, 1): 0,    (1, 2): 0,
            (2, 0): 500,  (2, 1): 0,    (2, 2): 0,
        }
        return cls(
            costs=costs,
            review_cost=5,
            class_names=["no_defect", "low_severity", "high_severity"],
        )

    @classmethod
    def from_yaml(cls, path: str) -> CostMatrix:
        """Load cost matrix from YAML config."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        costs = {}
        for key, val in cfg["costs"].items():
            parts = [int(x) for x in key.split(",")]
            costs[tuple(parts)] = val
        return cls(
            costs=costs,
            review_cost=cfg["review_cost"],
            class_names=cfg.get("class_names", []),
        )


def compute_expected_cost(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    abstained: np.ndarray,
    cost_matrix: CostMatrix,
) -> dict:
    """Compute expected inspection cost.

    Args:
        true_labels: (N,) true class labels.
        predictions: (N,) predicted class, -1 where abstained.
        abstained: (N,) boolean mask.
        cost_matrix: CostMatrix instance.

    Returns:
        Dict with total_cost, cost_per_sample, cost_per_1000,
        cost_breakdown.
    """
    N = len(true_labels)
    total = 0.0
    breakdown = {"classification_cost": 0.0, "review_cost": 0.0}

    for i in range(N):
        if abstained[i]:
            total += cost_matrix.review_cost
            breakdown["review_cost"] += cost_matrix.review_cost
        else:
            key = (int(true_labels[i]), int(predictions[i]))
            c = cost_matrix.costs.get(key, 0.0)
            total += c
            breakdown["classification_cost"] += c

    return {
        "total_cost": total,
        "cost_per_sample": total / N,
        "cost_per_1000": total / N * 1000,
        "cost_breakdown": breakdown,
        "n_samples": N,
    }


def sweep_coverage_vs_cost(
    softmax_probs: np.ndarray,
    true_labels: np.ndarray,
    cost_matrix: CostMatrix,
    alphas: Optional[np.ndarray] = None,
) -> list:
    """Sweep coverage targets and compute cost at each.

    Splits input 50/50 into calibration and evaluation sets.

    Args:
        softmax_probs: (N, C) model softmax outputs.
        true_labels: (N,) true labels.
        cost_matrix: CostMatrix instance.
        alphas: miscoverage rates to sweep (default: 0.01 to 0.50).

    Returns:
        List of dicts per alpha with coverage, abstention_rate,
        effective_f1, cost_per_1000.
    """
    from .conformal import ConformalClassifier

    if alphas is None:
        alphas = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    N = len(true_labels)
    n_cal = N // 2
    cal_probs, eval_probs = softmax_probs[:n_cal], softmax_probs[n_cal:]
    cal_labels, eval_labels = true_labels[:n_cal], true_labels[n_cal:]

    results = []
    for alpha in alphas:
        cc = ConformalClassifier(alpha=float(alpha))
        cc.calibrate(cal_probs, cal_labels)
        eval_result = cc.evaluate(eval_probs, eval_labels)
        preds, abstained_mask = cc.predict_with_abstention(eval_probs)
        cost_result = compute_expected_cost(
            eval_labels, preds, abstained_mask, cost_matrix,
        )
        results.append({
            "alpha": float(alpha),
            "target_coverage": 1 - float(alpha),
            **eval_result,
            **cost_result,
        })

    return results
