"""Tests for evaluation metrics."""

import numpy as np
import pytest

from simtodata.evaluation.metrics import (
    compute_all_metrics,
    compute_auroc,
    compute_ece,
    compute_macro_f1,
    compute_per_class_metrics,
)


class TestMacroF1:
    def test_perfect(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert compute_macro_f1(y_true, y_pred) == 1.0

    def test_random(self):
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1, 2], size=1000)
        y_pred = rng.choice([0, 1, 2], size=1000)
        f1 = compute_macro_f1(y_true, y_pred)
        assert 0.1 < f1 < 0.6


class TestAUROC:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
        ])
        assert compute_auroc(y_true, y_proba) > 0.99


class TestECE:
    def test_perfect_calibration(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        ece = compute_ece(y_true, y_proba)
        assert ece < 0.01

    def test_overconfident(self):
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_proba = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
            [0.99, 0.005, 0.005],  # wrong but overconfident
            [0.005, 0.99, 0.005],  # wrong but overconfident
            [0.005, 0.005, 0.99],
        ])
        ece = compute_ece(y_true, y_proba)
        assert ece > 0

    def test_no_division_by_zero(self):
        y_true = np.array([0, 0, 0])
        y_proba = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        ece = compute_ece(y_true, y_proba)
        assert np.isfinite(ece)


class TestPerClassMetrics:
    def test_structure(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        result = compute_per_class_metrics(y_true, y_pred)
        assert "precision" in result
        assert "recall" in result
        assert len(result["precision"]) == 3


class TestComputeAllMetrics:
    def test_all_keys_present(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        assert "macro_f1" in metrics
        assert "auroc" in metrics
        assert "ece" in metrics
        assert "per_class" in metrics
