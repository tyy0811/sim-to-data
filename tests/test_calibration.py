"""Tests for calibration analysis."""

import numpy as np
import pytest

from simtodata.evaluation.calibration import reliability_diagram


class TestReliabilityDiagram:
    def test_bin_counts_sum_to_n(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        _, _, counts = reliability_diagram(y_true, y_proba, n_bins=5)
        assert counts.sum() == len(y_true)

    def test_perfect_calibration(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        confidences, accuracies, _ = reliability_diagram(y_true, y_proba, n_bins=5)
        for c, a in zip(confidences, accuracies):
            if not np.isnan(c):
                assert abs(a - 1.0) < 0.01  # perfect accuracy

    def test_output_shapes(self):
        y_true = np.random.choice([0, 1, 2], 100)
        y_proba = np.random.dirichlet([1, 1, 1], 100)
        c, a, n = reliability_diagram(y_true, y_proba, n_bins=10)
        assert len(c) == 10
        assert len(a) == 10
        assert len(n) == 10
