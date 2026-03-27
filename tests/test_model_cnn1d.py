"""Tests for the 1D CNN defect classifier."""

import torch
import pytest

from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.factory import model_from_config


class TestDefectCNN1D:
    def test_output_shape(self):
        model = DefectCNN1D()
        x = torch.randn(16, 1, 1024)
        out = model(x)
        assert out.shape == (16, 3)

    def test_single_sample(self):
        model = DefectCNN1D()
        x = torch.randn(1, 1, 1024)
        out = model(x)
        assert out.shape == (1, 3)

    def test_large_batch(self):
        model = DefectCNN1D()
        x = torch.randn(256, 1, 1024)
        out = model(x)
        assert out.shape == (256, 3)

    def test_gradients_flow(self):
        model = DefectCNN1D()
        x = torch.randn(4, 1, 1024)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_param_count(self):
        model = DefectCNN1D()
        count = model.param_count()
        assert 20_000 < count < 200_000


class TestFactory:
    def test_from_config(self):
        model = model_from_config("configs/model_cnn1d.yaml")
        assert isinstance(model, DefectCNN1D)
        x = torch.randn(2, 1, 1024)
        out = model(x)
        assert out.shape == (2, 3)
