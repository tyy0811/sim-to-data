"""Noise injection functions for simulating sensor degradation."""

import numpy as np


def add_gaussian_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise at a specified SNR level."""
    signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), len(signal))
    return signal + noise


def add_baseline_drift(signal: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Add low-frequency sinusoidal baseline drift."""
    if amplitude == 0:
        return signal.copy()
    freq = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0, 2.0 * np.pi)
    t = np.linspace(0, 1, len(signal))
    drift = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return signal + drift


def add_temporal_jitter(signal: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Shift signal in time with zero-padding (no circular wrap)."""
    if max_shift == 0:
        return signal.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return signal.copy()
    result = np.zeros_like(signal)
    if shift > 0:
        result[shift:] = signal[:-shift]
    else:
        result[:shift] = signal[-shift:]
    return result


def add_masked_dropout(
    signal: np.ndarray, n_gaps: int, gap_length_range: tuple, rng: np.random.Generator
) -> np.ndarray:
    """Zero out contiguous windows to simulate coupling loss or acquisition gaps."""
    if n_gaps == 0:
        return signal.copy()
    result = signal.copy()
    length = len(signal)
    for _ in range(n_gaps):
        gap_len = int(rng.integers(gap_length_range[0], gap_length_range[1] + 1))
        start = int(rng.integers(0, max(1, length - gap_len)))
        result[start : start + gap_len] = 0.0
    return result


def add_gain_variation(signal: np.ndarray, gain_factor: float) -> np.ndarray:
    """Apply amplitude scaling to simulate sensor gain variation."""
    return signal * gain_factor


def apply_all_noise(
    signal: np.ndarray,
    params,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply all noise types in sequence: gaussian -> drift -> gain -> jitter -> dropout.

    Args:
        signal: Clean input signal.
        params: TraceParams (or any object with snr_db, baseline_drift_amplitude,
                gain_variation, jitter_samples, n_dropout_gaps, dropout_gap_length).
        rng: NumPy random generator.
    """
    result = add_gaussian_noise(signal, params.snr_db, rng)
    result = add_baseline_drift(result, params.baseline_drift_amplitude, rng)
    result = add_gain_variation(result, params.gain_variation)
    result = add_temporal_jitter(result, params.jitter_samples, rng)
    result = add_masked_dropout(result, params.n_dropout_gaps, params.dropout_gap_length, rng)
    return result
