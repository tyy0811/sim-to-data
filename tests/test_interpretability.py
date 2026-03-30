"""Tests for Grad-CAM interpretability functions."""

import torch

from simtodata.evaluation.interpretability import gradcam_1d
from simtodata.models.cnn1d import DefectCNN1D


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


from simtodata.evaluation.interpretability import compute_attribution_batch


class TestComputeAttributionBatch:
    def test_returns_expected_keys(self):
        model, layer = _get_model_and_layer()
        # Create a tiny fake dataset
        from torch.utils.data import TensorDataset
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
        from torch.utils.data import TensorDataset
        signals = torch.randn(10, 1, 1024)
        labels = torch.randint(0, 3, (10,))
        dataset = TensorDataset(signals, labels)
        result = compute_attribution_batch(model, dataset, layer, n_samples=5, seed=42)
        for i in range(5):
            cam = result["attributions"][i]
            assert cam.min() >= 0.0
            assert abs(cam.max() - 1.0) < 1e-6, f"Sample {i} max={cam.max()}"
