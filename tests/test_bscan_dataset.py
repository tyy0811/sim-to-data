"""Tests for B-scan dataset and resize utility."""

import numpy as np
import torch

import pytest

from simtodata.data.bscan_dataset import BscanDataset, resize_bscan


class TestResizeBscan:
    def test_output_shape(self):
        bscan = np.random.randn(128, 512).astype(np.float32)
        resized = resize_bscan(bscan, target_shape=(64, 64))
        assert resized.shape == (64, 64)

    def test_preserves_dtype(self):
        bscan = np.random.randn(256, 256).astype(np.float32)
        resized = resize_bscan(bscan, target_shape=(64, 64))
        assert resized.dtype == np.float32

    def test_finite(self):
        bscan = np.random.randn(100, 200).astype(np.float32)
        resized = resize_bscan(bscan, target_shape=(64, 64))
        assert np.all(np.isfinite(resized))

    def test_identity_when_already_correct_shape(self):
        bscan = np.random.randn(64, 64).astype(np.float32)
        resized = resize_bscan(bscan, target_shape=(64, 64))
        np.testing.assert_array_equal(resized, bscan)


class TestBscanDataset:
    def test_tensor_shapes(self):
        bscans = np.random.randn(10, 64, 64).astype(np.float32)
        labels = np.array([0, 1] * 5, dtype=np.int64)
        ds = BscanDataset(bscans, labels)
        x, y = ds[0]
        assert x.shape == (1, 64, 64)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_length(self):
        bscans = np.random.randn(20, 64, 64).astype(np.float32)
        labels = np.zeros(20, dtype=np.int64)
        ds = BscanDataset(bscans, labels)
        assert len(ds) == 20

    def test_labels_preserved(self):
        bscans = np.random.randn(5, 64, 64).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
        ds = BscanDataset(bscans, labels)
        for i in range(5):
            _, y = ds[i]
            assert y.item() == labels[i]

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            BscanDataset(np.zeros((2, 64, 64)), np.zeros(1))
        with pytest.raises(ValueError, match="length mismatch"):
            BscanDataset(np.zeros((1, 64, 64)), np.zeros(2))
