"""Tests for Grad-CAM interpretability functions."""

import importlib.util
import logging
import pathlib

import torch
from torch.utils.data import TensorDataset

from simtodata.evaluation.interpretability import (
    compute_attribution_batch,
    gradcam_1d,
    gradcam_2d,
)
from simtodata.models.cnn1d import DefectCNN1D

# Import _find_sample from experiments script (not a package).
_fig_spec = importlib.util.spec_from_file_location(
    "generate_gradcam_figures",
    pathlib.Path(__file__).resolve().parents[1] / "experiments" / "generate_gradcam_figures.py",
)
_fig_mod = importlib.util.module_from_spec(_fig_spec)
_fig_spec.loader.exec_module(_fig_mod)
_find_sample = _fig_mod._find_sample


def _get_model_and_layer():
    """Random-weight model and its last Conv1d layer."""
    model = DefectCNN1D()
    model.eval()
    # features is Sequential: [Conv1d, BN, ReLU, MaxPool] x 4 + AdaptiveAvgPool
    # Last Conv1d is at index -5 (before final BN, ReLU, MaxPool, AvgPool)
    last_conv = None
    for module in model.features:
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module
    return model, last_conv


class TestGradcam1d:
    def test_output_shape(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam = gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert cam.shape == (1024,)

    def test_output_finite(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam = gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert all(c == c for c in cam)  # no NaN
        assert all(abs(c) < float("inf") for c in cam)  # no Inf

    def test_output_nonnegative(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam = gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert all(c >= 0 for c in cam)

    def test_different_classes_differ(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam0 = gradcam_1d(model, x, target_class=0, target_layer=layer)
        cam1 = gradcam_1d(model, x, target_class=1, target_layer=layer)
        # With random weights, different class gradients produce different maps
        assert not all(abs(a - b) < 1e-10 for a, b in zip(cam0, cam1))

    def test_hooks_cleaned_up(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert len(layer._forward_hooks) == 0
        assert len(layer._backward_hooks) == 0


class TestGradcam2d:
    @staticmethod
    def _make_tiny_2d_cnn():
        """Minimal 2D CNN for testing gradcam_2d."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 3),
        )
        model.eval()
        return model, model[0]  # model, target_layer (Conv2d)

    def test_output_shape(self):
        model, layer = self._make_tiny_2d_cnn()
        x = torch.randn(1, 1, 16, 16)
        cam = gradcam_2d(model, x, target_class=0, target_layer=layer)
        assert cam.shape == (16, 16)

    def test_output_nonnegative(self):
        model, layer = self._make_tiny_2d_cnn()
        x = torch.randn(1, 1, 16, 16)
        cam = gradcam_2d(model, x, target_class=0, target_layer=layer)
        assert cam.min() >= 0.0

    def test_hooks_cleaned_up(self):
        model, layer = self._make_tiny_2d_cnn()
        x = torch.randn(1, 1, 16, 16)
        gradcam_2d(model, x, target_class=0, target_layer=layer)
        assert len(layer._forward_hooks) == 0
        assert len(layer._backward_hooks) == 0

    def test_different_classes_differ(self):
        model, layer = self._make_tiny_2d_cnn()
        x = torch.randn(1, 1, 16, 16)
        cam0 = gradcam_2d(model, x, target_class=0, target_layer=layer)
        cam1 = gradcam_2d(model, x, target_class=1, target_layer=layer)
        assert not (cam0 == cam1).all()


class TestFindSampleFallback:
    """Regression test: _find_sample must warn when falling back."""

    def _make_biased_model_and_dataset(self):
        """Model that always predicts class 0, dataset with only class-2 samples.

        Because every severity=2 sample is predicted as 0, requesting
        correct=True for severity=2 is impossible, forcing the fallback.
        """
        model = DefectCNN1D()
        model.eval()
        # Overwrite classifier bias so class-0 logit dominates
        with torch.no_grad():
            model.classifier[-1].bias.fill_(0.0)
            model.classifier[-1].bias[0] = 100.0

        last_conv = None
        for module in model.features:
            if isinstance(module, torch.nn.Conv1d):
                last_conv = module

        signals = torch.randn(5, 1, 1024)
        labels = torch.full((5,), 2)  # all severity=2
        dataset = TensorDataset(signals, labels)
        return model, last_conv, dataset

    def test_warns_on_fallback(self, caplog):
        model, layer, dataset = self._make_biased_model_and_dataset()
        with caplog.at_level(logging.WARNING, logger="generate_gradcam_figures"):
            result = _find_sample(
                dataset, severity=2, model=model,
                target_layer=layer, correct=True,
            )
        assert result is not None, "fallback should still return a sample"
        assert any("falling back" in r.message for r in caplog.records)

    def test_no_warning_when_exact_match(self, caplog):
        model, layer, dataset = self._make_biased_model_and_dataset()
        # correct=False matches because model predicts 0 != 2
        with caplog.at_level(logging.WARNING, logger="generate_gradcam_figures"):
            result = _find_sample(
                dataset, severity=2, model=model,
                target_layer=layer, correct=False,
            )
        assert result is not None
        assert not any("falling back" in r.message for r in caplog.records)


class TestComputeAttributionBatch:
    def test_returns_expected_keys(self):
        model, layer = _get_model_and_layer()
        signals = torch.randn(10, 1, 1024)
        labels = torch.randint(0, 3, (10,))
        dataset = TensorDataset(signals, labels)
        result = compute_attribution_batch(model, dataset, layer, n_samples=5, seed=42)
        assert set(result.keys()) == {"signals", "attributions", "labels", "predictions"}
        assert result["signals"].shape == (5, 1024)
        assert result["attributions"].shape == (5, 1024)
        assert result["labels"].shape == (5,)
        assert result["predictions"].shape == (5,)

    def test_attributions_normalized(self):
        model, layer = _get_model_and_layer()
        signals = torch.randn(10, 1, 1024)
        labels = torch.randint(0, 3, (10,))
        dataset = TensorDataset(signals, labels)
        result = compute_attribution_batch(model, dataset, layer, n_samples=5, seed=42)
        for i in range(5):
            cam = result["attributions"][i]
            assert cam.min() >= 0.0
            # All-zero CAMs (ReLU killed everything) stay at 0; others peak at 1.
            if cam.max() > 0:
                assert abs(cam.max() - 1.0) < 1e-6, f"Sample {i} max={cam.max()}"
