"""Tests for dataset generation, loading, and transforms."""

import numpy as np
import os
import pytest
import tempfile

from simtodata.data.generate import generate_dataset
from simtodata.simulator.regime import RegimeConfig


def _source_regime():
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


class TestGenerateDataset:
    def test_shapes(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 100, seed=42, class_distribution=class_dist)
        assert data["signals"].shape == (100, 1024)
        assert data["labels"].shape == (100,)

    def test_labels_valid(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 300, seed=42, class_distribution=class_dist)
        assert set(np.unique(data["labels"])) == {0, 1, 2}

    def test_approximate_class_balance(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 3000, seed=42, class_distribution=class_dist)
        counts = np.bincount(data["labels"])
        for c in counts:
            assert abs(c / 3000 - 0.33) < 0.05

    def test_deterministic(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        d1 = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        d2 = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        np.testing.assert_array_equal(d1["signals"], d2["signals"])
        np.testing.assert_array_equal(d1["labels"], d2["labels"])

    def test_signals_finite(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        assert np.all(np.isfinite(data["signals"]))

    def test_save_and_load(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            np.savez(path, signals=data["signals"], labels=data["labels"])
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded["signals"], data["signals"])
            np.testing.assert_array_equal(loaded["labels"], data["labels"])
