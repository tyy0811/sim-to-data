"""PyTorch Dataset for B-scan images and resize utility."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def resize_bscan(bscan: np.ndarray, target_shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Resize a B-scan to target shape using bilinear interpolation.

    If already the correct shape, returns the input unchanged.
    """
    if bscan.shape == target_shape:
        return bscan
    factors = (target_shape[0] / bscan.shape[0], target_shape[1] / bscan.shape[1])
    return zoom(bscan, factors, order=1).astype(np.float32)


class BscanDataset(Dataset):
    """Dataset wrapping pre-resized B-scan arrays.

    Args:
        bscans: (N, H, W) float32 array of B-scan images (already resized).
        labels: (N,) int64 array of binary labels (0=no flaw, 1=flaw).
    """

    def __init__(self, bscans: np.ndarray, labels: np.ndarray):
        if len(bscans) != len(labels):
            raise ValueError(
                f"bscans and labels length mismatch: {len(bscans)} vs {len(labels)}"
            )
        self.bscans = bscans.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bscan = torch.from_numpy(self.bscans[idx]).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bscan, label
