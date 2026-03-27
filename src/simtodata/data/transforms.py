"""Signal transforms for preprocessing."""

import numpy as np
from scipy.signal import stft as scipy_stft
import torch


class Normalize:
    """Per-trace zero-mean, unit-variance normalization."""

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        mean = signal.mean()
        std = signal.std()
        if std > 0:
            return (signal - mean) / std
        return signal - mean


class SpectrogramTransform:
    """STFT spectrogram transform for 2D CNN input."""

    def __init__(self, window_size=64, hop_length=16, fs=50e6):
        self.window_size = window_size
        self.hop_length = hop_length
        self.fs = fs

    def __call__(self, signal):
        """Transform (1, 1024) tensor to (1, n_freq, n_time) spectrogram."""
        sig_np = signal.squeeze(0).numpy()
        _, _, Zxx = scipy_stft(sig_np, fs=self.fs, nperseg=self.window_size,
                               noverlap=self.window_size - self.hop_length)
        magnitude = np.log1p(np.abs(Zxx))
        # Normalize per-spectrogram
        mean = magnitude.mean()
        std = magnitude.std()
        if std > 0:
            magnitude = (magnitude - mean) / std
        spec_tensor = torch.from_numpy(magnitude.astype(np.float32)).unsqueeze(0)
        return spec_tensor
