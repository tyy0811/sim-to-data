"""PyTorch Dataset for inspection signals."""

import numpy as np
import torch
from torch.utils.data import Dataset


class InspectionDataset(Dataset):
    """Dataset wrapping .npz files of synthetic A-scan traces."""

    def __init__(self, npz_path: str, transform=None):
        data = np.load(npz_path)
        self.signals = data["signals"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).unsqueeze(0)  # (1, 1024)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            signal = self.transform(signal)
        return signal, label
