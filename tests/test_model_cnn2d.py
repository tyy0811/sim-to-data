"""Tests for 2D CNN spectrogram classifier."""

import torch
from simtodata.models.cnn2d_spectrogram import DefectCNN2D


class TestDefectCNN2D:
    def test_output_shape(self):
        model = DefectCNN2D()
        x = torch.randn(8, 1, 33, 61)
        out = model(x)
        assert out.shape == (8, 3)

    def test_gradients_flow(self):
        model = DefectCNN2D()
        x = torch.randn(4, 1, 33, 61)
        out = model(x)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient: {name}"

    def test_from_config(self):
        from simtodata.models.factory import model_from_config
        model = model_from_config("configs/model_spec2d.yaml")
        assert isinstance(model, DefectCNN2D)
