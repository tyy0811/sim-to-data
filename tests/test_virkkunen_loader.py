"""Tests for Virkkunen ML-NDT data loader.

Tests create small synthetic binary blobs mimicking the dataset format
so they run without requiring the actual dataset to be downloaded.
"""

import json
import numpy as np
import os
import pytest

from simtodata.data.virkkunen import VirkkunenLoader


def _make_fake_batch(tmpdir, uuid, n_samples=3, h=256, w=256):
    """Create a fake .bins/.labels/.jsons batch for testing."""
    # .bins: UInt16 binary, N samples of h*w
    data = np.random.randint(0, 65535, (n_samples, h, w), dtype=np.uint16)
    bins_path = os.path.join(tmpdir, f"{uuid}.bins")
    data.tofile(bins_path)

    # .labels: one line per sample, "label\tsize"
    labels_path = os.path.join(tmpdir, f"{uuid}.labels")
    with open(labels_path, "w") as f:
        for i in range(n_samples):
            label = i % 2  # alternate 0, 1
            f.write(f"{label}\t{0.5 * (i % 2)}\n")

    # .jsons: concatenated JSON objects (actual dataset format)
    jsons_path = os.path.join(tmpdir, f"{uuid}.jsons")
    with open(jsons_path, "w") as f:
        for i in range(n_samples):
            json.dump({"sample": i, "flaw_size": 0.5 * (i % 2)}, f)

    return data


class TestVirkkunenLoader:
    def test_load_batch_shapes(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch01", n_samples=4)
        loader = VirkkunenLoader(str(tmp_path))
        bscans, labels, metadata = loader.load_batch("batch01")
        assert bscans.shape == (4, 256, 256)
        assert len(labels) == 4
        assert len(metadata) == 4
        assert bscans.dtype == np.float32

    def test_load_batch_labels_binary(self, tmp_path):
        _make_fake_batch(str(tmp_path), "batch02", n_samples=6)
        loader = VirkkunenLoader(str(tmp_path))
        _, labels, _ = loader.load_batch("batch02")
        assert all(lab in (0, 1) for lab in labels)

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

    def test_load_all_empty_dir_raises(self, tmp_path):
        loader = VirkkunenLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No .bins batch files"):
            loader.load_all()

    def test_missing_labels_raises(self, tmp_path):
        """A .bins without a matching .labels should fail fast."""
        data = np.zeros((2, 256, 256), dtype=np.uint16)
        data.tofile(str(tmp_path / "orphan.bins"))
        loader = VirkkunenLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Missing labels file"):
            loader.load_batch("orphan")

    def test_metadata_count_aligned(self, tmp_path):
        """All three arrays should be truncated to the shortest."""
        _make_fake_batch(str(tmp_path), "short_meta", n_samples=3)
        # Overwrite jsons with only 2 entries
        jsons_path = os.path.join(str(tmp_path), "short_meta.jsons")
        with open(jsons_path, "w") as f:
            json.dump({"sample": 0}, f)
            json.dump({"sample": 1}, f)
        loader = VirkkunenLoader(str(tmp_path))
        bscans, labels, metadata = loader.load_batch("short_meta")
        assert len(bscans) == len(labels) == len(metadata) == 2
