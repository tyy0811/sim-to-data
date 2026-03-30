"""Conformal prediction for multi-class defect classification.

Provides distribution-free coverage guarantees with selective prediction:
classify when confident, abstain when uncertain (flag for human review).

Uses Adaptive Prediction Sets (APS) nonconformity scores, which produce
smaller prediction sets than the simpler 1-p(true) threshold method.

Based on:
- Romano, Sesia & Candes (2020). Classification with Valid Adaptive
  Prediction Sets.
- Angelopoulos & Bates (2021). A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification.
"""

from __future__ import annotations

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Exact order statistic for finite-sample conformal guarantee.

    Uses k = ceil((N+1)(1-alpha)) without interpolation.
    np.quantile interpolates by default, violating the finite-sample bound.

    Args:
        scores: (N,) nonconformity scores from calibration set.
        alpha: miscoverage rate (0.05 = 95% coverage).

    Returns:
        Calibrated threshold q_hat.
    """
    N = len(scores)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    sorted_scores = np.sort(scores)
    if k > N:
        return float("inf")
    return float(sorted_scores[k - 1])


class ConformalClassifier:
    """Conformal selective prediction for safety-critical classification.

    Workflow:
        1. calibrate(cal_softmax, cal_labels) — set threshold q_hat
        2. predict_with_abstention(test_softmax) — classify or abstain

    Prediction set has 1 class = classify.
    Prediction set has >1 class = abstain (flag for human review).

    Coverage guarantee: P(true class in prediction set) >= 1 - alpha,
    regardless of the data distribution or model quality.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None
        self.n_classes = None

    def calibrate(self, cal_softmax: np.ndarray, cal_labels: np.ndarray) -> float:
        """Calibrate on held-out data using APS nonconformity scores.

        APS score: sort classes by descending probability, walk down
        accumulating probability, score = cumulative when true class
        is reached. High score = true class ranked low = poor confidence.

        Args:
            cal_softmax: (N, C) softmax probabilities.
            cal_labels: (N,) integer true class labels.

        Returns:
            q_hat: calibrated APS threshold.
        """
        N, C = cal_softmax.shape
        self.n_classes = C
        scores = np.empty(N)

        for i in range(N):
            sorted_indices = np.argsort(-cal_softmax[i])
            cumsum = 0.0
            for idx in sorted_indices:
                cumsum += cal_softmax[i, idx]
                if idx == cal_labels[i]:
                    scores[i] = cumsum
                    break

        self.q_hat = _conformal_quantile(scores, self.alpha)
        return self.q_hat

    def predict_sets(self, test_softmax: np.ndarray) -> list:
        """Return prediction sets for each sample.

        Walk classes by descending probability, accumulate until
        cumulative probability >= q_hat.

        Args:
            test_softmax: (B, C) softmax probabilities.

        Returns:
            List of sets, each containing class indices.
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict_sets()")

        pred_sets = []
        for i in range(test_softmax.shape[0]):
            sorted_indices = np.argsort(-test_softmax[i])
            cumsum = 0.0
            pset = set()
            for idx in sorted_indices:
                cumsum += test_softmax[i, idx]
                pset.add(int(idx))
                if cumsum >= self.q_hat:
                    break
            pred_sets.append(pset)
        return pred_sets

    def predict_with_abstention(self, test_softmax: np.ndarray):
        """Classify or abstain based on prediction set size.

        Args:
            test_softmax: (B, C) softmax probabilities.

        Returns:
            predictions: (B,) predicted class, -1 where abstained.
            abstained: (B,) boolean mask, True where abstained.
        """
        pred_sets = self.predict_sets(test_softmax)
        B = test_softmax.shape[0]
        predictions = np.full(B, -1, dtype=int)
        abstained = np.zeros(B, dtype=bool)

        for i, pset in enumerate(pred_sets):
            if len(pset) == 1:
                predictions[i] = next(iter(pset))
            else:
                abstained[i] = True

        return predictions, abstained

    def evaluate(self, test_softmax: np.ndarray, test_labels: np.ndarray) -> dict:
        """Full evaluation: coverage, abstention, effective metrics.

        Args:
            test_softmax: (B, C) softmax probabilities.
            test_labels: (B,) true labels.

        Returns:
            Dict with coverage, abstention_rate, effective_f1,
            class_abstention_rates, n_evaluated, n_abstained.
        """
        from sklearn.metrics import f1_score

        pred_sets = self.predict_sets(test_softmax)
        predictions, abstained = self.predict_with_abstention(test_softmax)

        B = len(test_labels)
        n_abstained = int(abstained.sum())
        n_evaluated = B - n_abstained

        covered = sum(1 for i in range(B) if test_labels[i] in pred_sets[i])
        coverage = covered / B

        if n_evaluated > 0:
            mask = ~abstained
            effective_f1 = float(f1_score(
                test_labels[mask], predictions[mask],
                average="macro", zero_division=0.0,
            ))
        else:
            effective_f1 = float("nan")

        class_abstention = {}
        for c in range(self.n_classes):
            class_mask = test_labels == c
            if class_mask.sum() > 0:
                class_abstention[c] = float(abstained[class_mask].mean())

        return {
            "coverage": coverage,
            "abstention_rate": n_abstained / B,
            "effective_f1": effective_f1,
            "n_evaluated": n_evaluated,
            "n_abstained": n_abstained,
            "n_total": B,
            "class_abstention_rates": class_abstention,
            "q_hat": self.q_hat,
            "alpha": self.alpha,
        }
