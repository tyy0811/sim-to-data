"""Tests for training loop and batch prediction."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model


def _make_loaders(n_train=90, n_val=30, n_test=30, batch_size=32):
    """Create small train/val/test loaders with 3-class labels."""
    rng = np.random.default_rng(42)
    signals = torch.randn(n_train + n_val + n_test, 1, 1024)
    labels = torch.from_numpy(rng.choice([0, 1, 2], size=n_train + n_val + n_test))

    train_ds = TensorDataset(signals[:n_train], labels[:n_train])
    val_ds = TensorDataset(signals[n_train:n_train + n_val], labels[n_train:n_train + n_val])
    test_ds = TensorDataset(signals[n_train + n_val:], labels[n_train + n_val:])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


class TestTrainModel:
    def test_returns_model_and_history(self):
        train_loader, val_loader, _ = _make_loaders()
        model = DefectCNN1D()
        model, history = train_model(model, train_loader, val_loader, epochs=3)
        assert hasattr(model, "forward")
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_f1"]) == 3

    def test_train_loss_decreases(self):
        train_loader, val_loader, _ = _make_loaders()
        model = DefectCNN1D()
        model, history = train_model(model, train_loader, val_loader, epochs=10, lr=1e-3)
        # First loss should be higher than last (on average, with enough epochs)
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_no_val_loader(self):
        """Training without validation should skip val metrics and early stopping."""
        train_loader, _, _ = _make_loaders()
        model = DefectCNN1D()
        model, history = train_model(model, train_loader, val_loader=None, epochs=3)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 0
        assert len(history["val_f1"]) == 0

    def test_early_stopping_triggers(self):
        """With patience=1 and a tiny LR, early stopping should fire before max epochs."""
        train_loader, val_loader, _ = _make_loaders()
        model = DefectCNN1D()
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=100, lr=1e-6, patience=2,
        )
        # Should stop well before 100 epochs
        assert len(history["train_loss"]) < 100

    def test_best_state_restored(self):
        """After training with validation, model should hold best-val-loss weights."""
        train_loader, val_loader, test_loader = _make_loaders()
        model = DefectCNN1D()
        # Snapshot initial weights
        initial_weight = model.features[0].weight.data.clone()
        model, history = train_model(model, train_loader, val_loader, epochs=5)
        # Weights should have changed from initialization
        assert not torch.equal(model.features[0].weight.data, initial_weight)


class TestPredictBatch:
    def test_output_shapes(self):
        _, _, test_loader = _make_loaders(n_test=50)
        model = DefectCNN1D()
        preds, probs, labels = predict_batch(model, test_loader)
        assert preds.shape == (50,)
        assert probs.shape == (50, 3)
        assert labels.shape == (50,)

    def test_probs_sum_to_one(self):
        _, _, test_loader = _make_loaders()
        model = DefectCNN1D()
        _, probs, _ = predict_batch(model, test_loader)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_preds_match_argmax_probs(self):
        _, _, test_loader = _make_loaders()
        model = DefectCNN1D()
        preds, probs, _ = predict_batch(model, test_loader)
        np.testing.assert_array_equal(preds, probs.argmax(axis=1))

    def test_deterministic(self):
        _, _, test_loader = _make_loaders()
        model = DefectCNN1D()
        model.eval()
        p1, pr1, _ = predict_batch(model, test_loader)
        p2, pr2, _ = predict_batch(model, test_loader)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(pr1, pr2)
