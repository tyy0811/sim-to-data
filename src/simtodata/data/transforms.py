"""Signal transforms for preprocessing."""

import torch


class Normalize:
    """Per-trace zero-mean, unit-variance normalization."""

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        mean = signal.mean()
        std = signal.std()
        if std > 0:
            return (signal - mean) / std
        return signal - mean
