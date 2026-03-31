"""Tests for CORAL domain adaptation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from simtodata.adaptation.coral import FeatureExtractor, coral_loss, train_with_coral


class _TinyModel(nn.Module):
    """Minimal model with a hookable 'features' layer."""

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(10, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _make_loaders(domain_shift=0.0, n=32, batch_size=16):
    src_x = torch.randn(n, 10)
    src_y = torch.randint(0, 3, (n,))
    tgt_x = torch.randn(n, 10) + domain_shift
    tgt_y = torch.randint(0, 3, (n,))
    src_loader = DataLoader(TensorDataset(src_x, src_y), batch_size=batch_size)
    tgt_loader = DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=batch_size)
    return src_loader, tgt_loader


class TestCoralLoss:
    def test_zero_for_same_features(self):
        """CORAL loss should be zero when source and target match."""
        x = torch.randn(32, 64)
        loss = coral_loss(x, x.clone())
        assert loss.item() < 1e-6

    def test_positive_for_different(self):
        """CORAL loss should be positive for different distributions."""
        src = torch.randn(32, 64)
        tgt = torch.randn(32, 64) + 2.0
        loss = coral_loss(src, tgt)
        assert loss.item() > 0

    def test_output_is_scalar(self):
        """CORAL loss should be a scalar."""
        loss = coral_loss(torch.randn(16, 32), torch.randn(16, 32))
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """CORAL loss should allow gradient computation."""
        src = torch.randn(16, 32, requires_grad=True)
        tgt = torch.randn(16, 32)
        loss = coral_loss(src, tgt)
        loss.backward()
        assert src.grad is not None
        assert torch.all(torch.isfinite(src.grad))


class TestFeatureExtractor:
    def test_returns_none_before_forward(self):
        """Should return None before any forward pass."""
        model = _TinyModel()
        extractor = FeatureExtractor(model, "features")
        assert extractor.get() is None

    def test_captures_features(self):
        """Should capture layer output after forward pass."""
        model = _TinyModel()
        extractor = FeatureExtractor(model, "features")
        x = torch.randn(4, 10)
        model(x)
        feats = extractor.get()
        assert feats is not None
        assert feats.shape == (4, 8)

    def test_gradients_flow_through_hook(self):
        """Hooked features must retain grad graph — regression test for detach() bug."""
        model = _TinyModel()
        extractor = FeatureExtractor(model, "features")
        x = torch.randn(4, 10)
        model(x)
        feats = extractor.get()
        loss = feats.sum()
        loss.backward()
        assert model.features.weight.grad is not None


class TestTrainWithCoral:
    def test_returns_epoch_history(self):
        """Should return per-epoch dicts with ce_loss and coral_loss."""
        model = _TinyModel()
        src_loader, tgt_loader = _make_loaders()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        history = train_with_coral(
            model, src_loader, tgt_loader, optimizer,
            feature_layer="features", n_epochs=2,
        )
        assert len(history) == 2
        for entry in history:
            assert "epoch" in entry
            assert "ce_loss" in entry
            assert "coral_loss" in entry

    def test_updates_parameters(self):
        """Training should actually change model weights."""
        model = _TinyModel()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        src_loader, tgt_loader = _make_loaders()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_with_coral(
            model, src_loader, tgt_loader, optimizer,
            feature_layer="features", n_epochs=3,
        )
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in model.named_parameters()
        )
        assert changed, "Model parameters should change after training"

    def test_coral_loss_nonzero_with_shift(self):
        """CORAL loss should be non-zero when domains differ."""
        model = _TinyModel()
        src_loader, tgt_loader = _make_loaders(domain_shift=5.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        history = train_with_coral(
            model, src_loader, tgt_loader, optimizer,
            feature_layer="features", n_epochs=1,
        )
        assert history[0]["coral_loss"] > 0
