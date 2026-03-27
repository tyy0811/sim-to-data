"""Tests for sklearn baseline classifiers."""

import numpy as np
import pytest

from simtodata.models.baselines import create_baseline


class TestBaselines:
    @pytest.fixture
    def train_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 11))
        y = rng.choice([0, 1, 2], size=200)
        return X, y

    def test_logistic_regression_fits(self, train_data):
        clf = create_baseline("logistic_regression")
        clf.fit(*train_data)
        preds = clf.predict(train_data[0][:10])
        assert preds.shape == (10,)
        assert all(p in {0, 1, 2} for p in preds)

    def test_gradient_boosting_fits(self, train_data):
        clf = create_baseline("gradient_boosting")
        clf.fit(*train_data)
        preds = clf.predict(train_data[0][:10])
        assert preds.shape == (10,)

    def test_predict_proba_valid(self, train_data):
        clf = create_baseline("gradient_boosting")
        clf.fit(*train_data)
        probs = clf.predict_proba(train_data[0][:10])
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0)

    def test_above_random(self, train_data):
        clf = create_baseline("gradient_boosting")
        clf.fit(*train_data)
        preds = clf.predict(train_data[0])
        accuracy = np.mean(preds == train_data[1])
        assert accuracy > 0.33
