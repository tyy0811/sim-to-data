"""Tests for hand-crafted feature extraction."""

import numpy as np

from simtodata.features.extract import extract_features, extract_features_batch
from simtodata.simulator.forward_model import TraceParams, generate_trace


class TestExtractFeatures:
    def test_output_length(self):
        signal = np.random.randn(1024)
        features = extract_features(signal)
        assert features.shape == (11,)

    def test_finite(self):
        signal = np.random.randn(1024)
        features = extract_features(signal)
        assert np.all(np.isfinite(features))

    def test_deterministic(self):
        signal = np.random.randn(1024)
        f1 = extract_features(signal)
        f2 = extract_features(signal)
        np.testing.assert_array_equal(f1, f2)

    def test_defect_has_more_peaks(self):
        params_no = TraceParams(
            thickness_mm=20.0, velocity_ms=6000.0, attenuation_np_mm=0.02,
            center_freq_mhz=3.0, pulse_sigma_us=1.0, has_defect=False,
        )
        params_def = TraceParams(
            thickness_mm=20.0, velocity_ms=6000.0, attenuation_np_mm=0.02,
            center_freq_mhz=3.0, pulse_sigma_us=1.0, has_defect=True,
            defect_depth_mm=10.0, defect_reflectivity=0.6,
        )
        f_no = extract_features(generate_trace(params_no))
        f_def = extract_features(generate_trace(params_def))
        assert f_def[0] >= f_no[0]  # n_peaks


class TestExtractFeaturesBatch:
    def test_batch_shape(self):
        signals = np.random.randn(20, 1024)
        features = extract_features_batch(signals)
        assert features.shape == (20, 11)
