"""Tests for Virkkunen ML-NDT data loader.

Tests create small synthetic binary blobs mimicking the dataset format
so they run without requiring the actual dataset to be downloaded.
"""

import json
import numpy as np
import os
import pytest

from simtodata.data.virkkunen import VirkkunenLoader


def _make_fake_batch(tmpdir, uuid, n_samples=3, shape=(256, 256, 100)):
    """Create a fake .bins/.labels/.jsons batch for testing."""
    time, spatial, channels = shape

    # .bins: UInt16 binary
    data = np.random.randint(0, 65535, (n_samples, channels, spatial, time), dtype=np.uint16)
    bins_path = os.path.join(tmpdir, f"{uuid}.bins")
    data.tofile(bins_path)

    # .labels: one line per sample, "label\tinfo"
    labels_path = os.path.join(tmpdir, f"{uuid}.labels")
    with open(labels_path, "w") as f:
        for i in range(n_samples):
            label = i % 2  # alternate 0, 1
            f.write(f"{label}\tinfo\n")

    # .jsons: list of metadata dicts
    jsons_path = os.path.join(tmpdir, f"{uuid}.jsons")
    metadata = [{"sample": i, "flaw_size": 0.5 * (i % 2)} for i in range(n_samples)]
    with open(jsons_path, "w") as f:
        json.dump(metadata, f)

    return data


class TestVirkkunenLoader:
    def test_load_batch_shapes(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch01", n_samples=4)
        loader = VirkkunenLoader(str(tmp_path))
        bscans, labels, metadata = loader.load_batch("batch01")
        assert bscans.shape == (4, 256, 256)  # one channel extracted
        assert len(labels) == 4
        assert bscans.dtype == np.float32

    def test_load_batch_labels_binary(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch02", n_samples=6)
        loader = VirkkunenLoader(str(tmp_path))
        _, labels, _ = loader.load_batch("batch02")
        assert all(l in (0, 1) for l in labels)

    def test_load_batch_normalized(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch03", n_samples=3)
        loader = VirkkunenLoader(str(tmp_path))
        bscans, _, _ = loader.load_batch("batch03")
        for i in range(len(bscans)):
            assert abs(bscans[i].mean()) < 0.1  # approximately zero-mean
            assert abs(bscans[i].std() - 1.0) < 0.1  # approximately unit-var

    def test_load_all_concatenates(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch_a", n_samples=3)
        _make_fake_batch(str(tmp_path), "batch_b", n_samples=5)
        loader = VirkkunenLoader(str(tmp_path))
        bscans, labels = loader.load_all()
        assert bscans.shape[0] == 8
        assert labels.shape[0] == 8

    def test_load_all_finite(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch_fin", n_samples=3)
        loader = VirkkunenLoader(str(tmp_path))
        bscans, _ = loader.load_all()
        assert np.all(np.isfinite(bscans))

    def test_channel_selection(self, tmp_path):
        """Verify that a specific channel can be extracted."""
        data = _make_fake_batch(str(tmp_path), "batch_ch", n_samples=2)
        loader = VirkkunenLoader(str(tmp_path), channel=5)
        bscans, _, _ = loader.load_batch("batch_ch")
        # Channel 5 data, normalized — just check shape is correct
        assert bscans.shape == (2, 256, 256)
