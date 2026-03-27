"""Tests for signal transforms."""

import numpy as np
import torch
import pytest

from simtodata.data.transforms import Normalize, SpectrogramTransform


class TestSpectrogramTransform:
    def test_output_shape(self):
        signal = torch.randn(1, 1024)
        transform = SpectrogramTransform(window_size=64, hop_length=16)
        spec = transform(signal)
        assert spec.dim() == 3  # (1, n_freq, n_time)
        assert spec.shape[0] == 1

    def test_finite(self):
        signal = torch.randn(1, 1024)
        transform = SpectrogramTransform()
        spec = transform(signal)
        assert torch.all(torch.isfinite(spec))

    def test_deterministic(self):
        signal = torch.randn(1, 1024)
        transform = SpectrogramTransform()
        s1 = transform(signal)
        s2 = transform(signal)
        torch.testing.assert_close(s1, s2)
