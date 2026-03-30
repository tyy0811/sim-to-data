"""Loader for the Virkkunen ML-NDT phased-array weld inspection dataset.

Dataset: https://github.com/iikka-v/ML-NDT
Paper: https://arxiv.org/abs/1903.11399
License: LGPL-3.0

Each batch file contains:
  - .bins: UInt16 binary, N samples of 256x256 (raw format: "UInt16, 256 x 256 x N")
  - .labels: N lines, tab-separated: flaw_label (0/1) <tab> equivalent_flaw_size
  - .jsons: N concatenated JSON objects with flaw metadata
"""

from __future__ import annotations

import json
import os
import re

import numpy as np


class VirkkunenLoader:
    """Load and normalize the Virkkunen ML-NDT dataset.

    Each sample is a single-channel 256x256 B-scan image (phased-array
    ultrasonic sector scan). Normalization is per-sample zero-mean,
    unit-variance.

    Args:
        data_dir: Directory containing .bins/.labels/.jsons files.
    """

    H = 256
    W = 256

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_batch(self, batch_uuid: str) -> tuple[np.ndarray, list[int], list[dict]]:
        """Load one batch file.

        Returns:
            bscans: (N, 256, 256) float32 array, zero-mean unit-variance per sample.
            labels: list of int (0=no flaw, 1=flaw).
            metadata: list of dicts from .jsons (best-effort; empty list on parse failure).
        """
        bins_path = os.path.join(self.data_dir, f"{batch_uuid}.bins")
        labels_path = os.path.join(self.data_dir, f"{batch_uuid}.labels")
        jsons_path = os.path.join(self.data_dir, f"{batch_uuid}.jsons")

        # Read binary: UInt16, each sample is H*W values
        raw = np.fromfile(bins_path, dtype=np.uint16)
        sample_size = self.H * self.W
        n_samples = len(raw) // sample_size
        bscans = raw[:n_samples * sample_size].reshape(
            n_samples, self.H, self.W,
        ).astype(np.float32)

        # Per-sample normalization: zero-mean, unit-variance
        for i in range(len(bscans)):
            std = bscans[i].std()
            bscans[i] = (bscans[i] - bscans[i].mean()) / (std + 1e-10)

        # Read labels: first column is 0/1 flaw indicator
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Missing labels file for batch {batch_uuid}: {labels_path}"
            )
        labels = []
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(int(line.split("\t")[0]))

        # Read metadata: concatenated JSON objects (not a JSON array)
        metadata = []
        if os.path.exists(jsons_path):
            try:
                with open(jsons_path) as f:
                    text = f.read()
                # Split concatenated JSON objects: insert comma between }{ pairs
                text = re.sub(r'\}\s*\{', '},{', text)
                metadata = json.loads(f'[{text}]')
            except (json.JSONDecodeError, ValueError):
                metadata = []

        # Align counts across all arrays
        n = min(len(bscans), len(labels))
        if metadata:
            n = min(n, len(metadata))
        else:
            metadata = [{}] * n
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

        if not batch_uuids:
            raise FileNotFoundError(
                f"No .bins batch files found in {self.data_dir}"
            )

        for uuid in sorted(batch_uuids):
            bscans, labels, _ = self.load_batch(uuid)
            all_bscans.append(bscans)
            all_labels.extend(labels)

        return np.concatenate(all_bscans), np.array(all_labels)
