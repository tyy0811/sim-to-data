"""Tests for CORAL domain adaptation."""

import torch

from simtodata.adaptation.coral import coral_loss


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
