"""Tests for conformal selective prediction."""

import numpy as np
import pytest

from simtodata.evaluation.conformal import (
    ConformalClassifier,
    _conformal_quantile,
)


def _softmax(x):
    """Numerically stable softmax for test data generation."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TestConformalQuantile:
    def test_exact_order_statistic_overflow(self):
        """k > N should return inf."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q = _conformal_quantile(scores, alpha=0.1)
        # k = ceil(6 * 0.9) = ceil(5.4) = 6 > N=5 → inf
        assert q == float("inf")

    def test_exact_order_statistic_value(self):
        """k <= N should return the k-th sorted value."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q = _conformal_quantile(scores, alpha=0.2)
        # k = ceil(6 * 0.8) = ceil(4.8) = 5, sorted[4] = 0.9
        assert q == 0.9


class TestConformalClassifier:
    def test_calibrate_sets_q_hat(self):
        np.random.seed(42)
        cc = ConformalClassifier(alpha=0.1)
        probs = _softmax(np.random.randn(100, 3))
        labels = np.random.randint(0, 3, size=100)
        q = cc.calibrate(probs, labels)
        assert q > 0
        assert cc.q_hat == q

    def test_coverage_guarantee_empirical(self):
        np.random.seed(42)
        N = 2000
        true = np.random.randint(0, 3, size=N)
        probs = np.eye(3)[true] * 0.7 + 0.1
        probs = probs / probs.sum(axis=1, keepdims=True)
        n_cal = N // 2
        cc = ConformalClassifier(alpha=0.1)
        cc.calibrate(probs[:n_cal], true[:n_cal])
        result = cc.evaluate(probs[n_cal:], true[n_cal:])
        assert result["coverage"] >= 0.89  # finite-sample slack

    def test_abstention_increases_with_coverage(self):
        np.random.seed(42)
        probs = _softmax(np.random.randn(500, 3))
        labels = np.random.randint(0, 3, size=500)
        cc90 = ConformalClassifier(alpha=0.10)
        cc95 = ConformalClassifier(alpha=0.05)
        cc90.calibrate(probs[:250], labels[:250])
        cc95.calibrate(probs[:250], labels[:250])
        r90 = cc90.evaluate(probs[250:], labels[250:])
        r95 = cc95.evaluate(probs[250:], labels[250:])
        assert r95["abstention_rate"] >= r90["abstention_rate"]

    def test_predict_with_abstention_output_format(self):
        cc = ConformalClassifier(alpha=0.1)
        cc.q_hat = 0.8
        cc.n_classes = 3
        probs = _softmax(np.random.randn(20, 3))
        preds, abstained = cc.predict_with_abstention(probs)
        assert preds.shape == (20,)
        assert abstained.shape == (20,)
        assert set(np.unique(abstained)).issubset({True, False})
        assert all(preds[abstained] == -1)

    def test_uncalibrated_raises(self):
        cc = ConformalClassifier(alpha=0.1)
        with pytest.raises(RuntimeError):
            cc.predict_sets(np.zeros((5, 3)))
