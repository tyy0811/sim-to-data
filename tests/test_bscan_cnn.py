"""Tests for B-scan 2D CNN."""

import torch
import numpy as np

from simtodata.models.cnn2d_bscan import BscanCNN


class TestBscanCNN:
    def test_output_shape(self):
        model = BscanCNN()
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, 2)

    def test_gradients_flow(self):
        model = BscanCNN()
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 1, 64, 64)

    def test_param_count_reasonable(self):
        model = BscanCNN()
        n = sum(p.numel() for p in model.parameters())
        assert 10_000 < n < 200_000  # should be ~30K

    def test_eval_mode_deterministic(self):
        model = BscanCNN()
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)
