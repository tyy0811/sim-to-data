"""Tests for dataset generation, loading, and transforms."""

import numpy as np
import os
import pytest
import tempfile

from simtodata.data.generate import generate_dataset


class TestGenerateDataset:
    def test_shapes(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 100, seed=42, class_distribution=class_dist)
        assert data["signals"].shape == (100, 1024)
        assert data["signals"].dtype == np.float32
        assert data["labels"].shape == (100,)

    def test_metadata_present(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        for key in ["thickness_mm", "velocity_ms", "attenuation_np_mm",
                     "defect_depth_mm", "defect_reflectivity", "snr_db"]:
            assert key in data, f"Missing metadata key: {key}"
            assert data[key].shape == (50,)
            assert data[key].dtype == np.float32

    def test_labels_valid(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 300, seed=42, class_distribution=class_dist)
        assert set(np.unique(data["labels"])) == {0, 1, 2}

    def test_approximate_class_balance(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 3000, seed=42, class_distribution=class_dist)
        counts = np.bincount(data["labels"])
        for c in counts:
            assert abs(c / 3000 - 0.33) < 0.05

    def test_deterministic(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        d1 = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        d2 = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        np.testing.assert_array_equal(d1["signals"], d2["signals"])
        np.testing.assert_array_equal(d1["labels"], d2["labels"])

    def test_signals_finite(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        assert np.all(np.isfinite(data["signals"]))

    def test_save_and_load(self, source_regime):
        regime = source_regime
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            np.savez(path, signals=data["signals"], labels=data["labels"])
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded["signals"], data["signals"])
            np.testing.assert_array_equal(loaded["labels"], data["labels"])


import torch
from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize


class TestInspectionDataset:
    def test_tensor_shapes(self, tmp_path):
        signals = np.random.randn(50, 1024).astype(np.float32)
        labels = np.array([0] * 17 + [1] * 17 + [2] * 16, dtype=np.int64)
        np.savez(tmp_path / "test.npz", signals=signals, labels=labels)
        ds = InspectionDataset(str(tmp_path / "test.npz"))
        signal, label = ds[0]
        assert signal.shape == (1, 1024)
        assert label.shape == ()
        assert signal.dtype == torch.float32

    def test_length(self, tmp_path):
        signals = np.random.randn(50, 1024).astype(np.float32)
        labels = np.zeros(50, dtype=np.int64)
        np.savez(tmp_path / "test.npz", signals=signals, labels=labels)
        ds = InspectionDataset(str(tmp_path / "test.npz"))
        assert len(ds) == 50

    def test_normalize_transform(self, tmp_path):
        signals = np.random.randn(10, 1024).astype(np.float32) * 5 + 3
        labels = np.zeros(10, dtype=np.int64)
        np.savez(tmp_path / "test.npz", signals=signals, labels=labels)
        ds = InspectionDataset(str(tmp_path / "test.npz"), transform=Normalize())
        signal, _ = ds[0]
        assert abs(signal.mean().item()) < 0.01
        assert abs(signal.std().item() - 1.0) < 0.01
