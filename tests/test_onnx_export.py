"""Tests for ONNX export and inference."""

import os

import numpy as np
import torch.nn as nn

from simtodata.export.onnx_export import export_to_onnx, verify_onnx


def _build_tiny_cnn():
    """Minimal CNN matching DefectCNN1D interface for testing."""
    return nn.Sequential(
        nn.Conv1d(1, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    )


class TestOnnxExport:
    def test_export_creates_file(self, tmp_path):
        model = _build_tiny_cnn()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path, trace_length=100)
        assert os.path.exists(path)

    def test_onnx_output_shape(self, tmp_path):
        model = _build_tiny_cnn()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path, trace_length=100)

        import onnxruntime as ort
        session = ort.InferenceSession(path)
        x = np.random.randn(4, 1, 100).astype(np.float32)
        out = session.run(None, {"trace": x})[0]
        assert out.shape == (4, 3)

    def test_onnx_matches_pytorch(self, tmp_path):
        model = _build_tiny_cnn()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path, trace_length=100)
        assert verify_onnx(model, path, trace_length=100, atol=1e-5)
