"""Hand-crafted feature extraction for baseline classifiers."""

import numpy as np
from scipy.signal import find_peaks


def extract_features(signal: np.ndarray, fs: float = 50e6) -> np.ndarray:
    """Extract 11 features from a single A-scan trace.

    Features:
        0: n_peaks
        1-3: peak_amplitudes_top3
        4-6: peak_times_top3
        7: signal_energy
        8: spectral_centroid
        9: mid_region_energy (5%-85% of trace)
        10: inter_peak_ratio
    """
    abs_signal = np.abs(signal)
    threshold = 0.1 * np.max(abs_signal) if np.max(abs_signal) > 0 else 0
    peaks, properties = find_peaks(abs_signal, height=threshold)
    heights = properties["peak_heights"] if len(peaks) > 0 else np.array([])
    sorted_idx = np.argsort(heights)[::-1]

    top_amps = np.zeros(3)
    top_times = np.zeros(3)
    for i in range(min(3, len(sorted_idx))):
        top_amps[i] = heights[sorted_idx[i]]
        top_times[i] = peaks[sorted_idx[i]]

    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)

    mid_start = int(0.05 * len(signal))
    mid_end = int(0.85 * len(signal))
    mid_energy = np.sum(signal[mid_start:mid_end] ** 2)

    inter_peak_ratio = top_amps[1] / (top_amps[0] + 1e-10)

    return np.array([
        len(peaks),
        *top_amps,
        *top_times,
        np.sum(signal**2),
        spectral_centroid,
        mid_energy,
        inter_peak_ratio,
    ])


def extract_features_batch(signals: np.ndarray, fs: float = 50e6) -> np.ndarray:
    """Extract features for a batch of signals."""
    return np.array([extract_features(s, fs) for s in signals])
