"""Tests for regime configuration and parameter sampling."""

import numpy as np
import pytest
import yaml

from simtodata.simulator.regime import RegimeConfig, load_regimes_from_yaml, sample_trace_params


def _source_regime():
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        defect_reflectivity=(0.1, 0.8),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


class TestRegimeConfig:
    def test_sample_params_in_range(self):
        regime = _source_regime()
        rng = np.random.default_rng(42)
        params = sample_trace_params(regime, rng, severity_class=1)
        assert 5800 <= params.velocity_ms <= 6200
        assert 0.01 <= params.attenuation_np_mm <= 0.05
        assert 2.0 <= params.center_freq_mhz <= 5.0
        assert 20.0 <= params.snr_db <= 40.0

    def test_deterministic(self):
        regime = _source_regime()
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        p1 = sample_trace_params(regime, r1, severity_class=1)
        p2 = sample_trace_params(regime, r2, severity_class=1)
        assert p1.velocity_ms == p2.velocity_ms
        assert p1.defect_depth_mm == p2.defect_depth_mm

    def test_no_defect_class(self):
        regime = _source_regime()
        rng = np.random.default_rng(42)
        params = sample_trace_params(regime, rng, severity_class=0)
        assert params.has_defect is False
        assert params.severity_label == 0

    def test_defect_depth_within_thickness(self):
        regime = _source_regime()
        rng = np.random.default_rng(42)
        for _ in range(100):
            params = sample_trace_params(regime, rng, severity_class=2)
            assert params.defect_depth_mm < params.thickness_mm

    def test_reflectivity_uses_regime_range(self):
        """Shifted regime has wider reflectivity range than module defaults."""
        regime = RegimeConfig(
            name="custom",
            thickness_mm=(15.0, 25.0),
            velocity_ms=(5800.0, 6200.0),
            attenuation_np_mm=(0.01, 0.05),
            center_freq_mhz=(2.0, 5.0),
            pulse_sigma_us=(0.5, 1.5),
            defect_depth_mm=(2.0, 20.0),
            defect_reflectivity=(0.05, 0.9),
            snr_db=(20.0, 40.0),
            baseline_drift=(0.0, 0.0),
            gain_variation=(1.0, 1.0),
            jitter_samples=(0, 0),
            dropout_n_gaps=(0, 0),
            dropout_gap_length=(0, 0),
        )
        rng = np.random.default_rng(42)
        low_reflectivities = []
        high_reflectivities = []
        for _ in range(200):
            p = sample_trace_params(regime, rng, severity_class=1)
            low_reflectivities.append(p.defect_reflectivity)
            p = sample_trace_params(regime, rng, severity_class=2)
            high_reflectivities.append(p.defect_reflectivity)
        # Low severity should sample from [0.05, 0.4) — below HIGH threshold
        assert min(low_reflectivities) >= 0.05
        assert max(low_reflectivities) < 0.4
        # High severity should sample from [0.4, 0.9]
        assert min(high_reflectivities) >= 0.4
        assert max(high_reflectivities) <= 0.9

    def test_reflectivity_loaded_from_yaml(self):
        regimes = load_regimes_from_yaml("configs/simulator.yaml")
        assert regimes["source"].defect_reflectivity == (0.1, 0.8)
        assert regimes["shifted"].defect_reflectivity == (0.05, 0.9)


class TestLoadRegimes:
    def test_load_from_yaml(self):
        regimes = load_regimes_from_yaml("configs/simulator.yaml")
        assert "source" in regimes
        assert "shifted" in regimes
        assert "randomized" in regimes

    def test_shifted_wider_than_source(self):
        regimes = load_regimes_from_yaml("configs/simulator.yaml")
        assert regimes["shifted"].snr_db[0] < regimes["source"].snr_db[0]
        assert regimes["shifted"].velocity_ms[1] > regimes["source"].velocity_ms[1]
