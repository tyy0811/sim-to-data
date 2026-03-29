"""Loader for the Virkkunen ML-NDT phased-array weld inspection dataset.

Dataset: https://github.com/iikka-v/ML-NDT
Paper: https://arxiv.org/abs/1903.11399
License: LGPL-3.0

Each batch file contains:
  - .bins: UInt16 binary, each sample is (channels=100, spatial=256, time=256)
  - .labels: one line per sample, "label<tab>info"
  - .jsons: JSON array of per-sample metadata
"""

from __future__ import annotations

import json
import os

import numpy as np


class VirkkunenLoader:
    """Load and normalize the Virkkunen ML-NDT dataset.

    Extracts a single channel from the multi-channel phased-array data,
    producing (spatial, time) B-scan images.

    Args:
        data_dir: Directory containing .bins/.labels/.jsons files.
        channel: Which of the 100 channels to extract (default: 0).
    """

    CHANNELS = 100
    SPATIAL = 256
    TIME = 256

    def __init__(self, data_dir: str, channel: int = 0):
        if not 0 <= channel < self.CHANNELS:
            raise ValueError(
                f"channel must be in [0, {self.CHANNELS}), got {channel}"
            )
        self.data_dir = data_dir
        self.channel = channel

    def load_batch(self, batch_uuid: str) -> tuple[np.ndarray, list[int], list[dict]]:
        """Load one batch file.

        Returns:
            bscans: (N, 256, 256) float32 array, zero-mean unit-variance per sample.
            labels: list of int (0=no flaw, 1=flaw).
            metadata: list of dicts from .jsons.
        """
        bins_path = os.path.join(self.data_dir, f"{batch_uuid}.bins")
        labels_path = os.path.join(self.data_dir, f"{batch_uuid}.labels")
        jsons_path = os.path.join(self.data_dir, f"{batch_uuid}.jsons")

        # Read binary: UInt16, each sample is (channels, spatial, time)
        raw = np.fromfile(bins_path, dtype=np.uint16)
        sample_size = self.CHANNELS * self.SPATIAL * self.TIME
        n_samples = len(raw) // sample_size
        volumes = raw[:n_samples * sample_size].reshape(
            n_samples, self.CHANNELS, self.SPATIAL, self.TIME,
        )

        # Extract one channel -> (N, spatial, time)
        bscans = volumes[:, self.channel, :, :].astype(np.float32)

        # Per-sample normalization: zero-mean, unit-variance
        for i in range(len(bscans)):
            std = bscans[i].std()
            bscans[i] = (bscans[i] - bscans[i].mean()) / (std + 1e-10)

        # Read labels
        with open(labels_path) as f:
            label_lines = f.readlines()
        labels = [int(line.strip().split("\t")[0]) for line in label_lines if line.strip()]

        # Read metadata
        with open(jsons_path) as f:
            metadata = json.load(f)

        # Align counts across all three arrays
        n = min(len(bscans), len(labels), len(metadata))
        return bscans[:n], labels[:n], metadata[:n]

    def load_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all batch files and concatenate.

        Returns:
            bscans: (N_total, 256, 256) float32 array.
            labels: (N_total,) int array.
        """
        all_bscans, all_labels = [], []
        batch_uuids = set()
        for f in os.listdir(self.data_dir):
            if f.endswith(".bins"):
                batch_uuids.add(f.replace(".bins", ""))

        for uuid in sorted(batch_uuids):
            bscans, labels, _ = self.load_batch(uuid)
            all_bscans.append(bscans)
            all_labels.extend(labels)

        return np.concatenate(all_bscans), np.array(all_labels)
