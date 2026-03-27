"""Pulse-echo ultrasonic forward model for A-scan signal generation.

Generates synthetic 1D A-scan traces for a contact transducer in pulse-echo mode.
A_surface is fixed at 1.0; all other amplitudes are relative to it.
"""

from dataclasses import dataclass
import numpy as np

SURFACE_ECHO_OFFSET_US = 1.0


@dataclass
class TraceParams:
    """Parameters for generating a single pulse-echo A-scan trace."""

    thickness_mm: float
    velocity_ms: float
    attenuation_np_mm: float
    center_freq_mhz: float
    pulse_sigma_us: float
    has_defect: bool = False
    defect_depth_mm: float = 0.0
    defect_reflectivity: float = 0.0
    severity_label: int = 0
    snr_db: float = 40.0
    baseline_drift_amplitude: float = 0.0
    gain_variation: float = 1.0
    jitter_samples: int = 0
    n_dropout_gaps: int = 0
    dropout_gap_length: tuple = (0, 0)  # (min, max) range — gap lengths sampled inside apply_all_noise
    n_samples: int = 1024
    sampling_rate_mhz: float = 50.0


def generate_pulse(t: np.ndarray, f_center_mhz: float, sigma_us: float, t0_us: float) -> np.ndarray:
    """Generate a Gabor wavelet pulse centered at t0."""
    envelope = np.exp(-((t - t0_us) ** 2) / (2.0 * sigma_us**2))
    carrier = np.sin(2.0 * np.pi * f_center_mhz * (t - t0_us))
    return envelope * carrier


def compute_arrival_time(depth_mm: float, velocity_ms: float) -> float:
    """Compute two-way travel time in microseconds."""
    velocity_mm_per_us = velocity_ms * 1e-3
    return 2.0 * depth_mm / velocity_mm_per_us


def compute_amplitude(initial: float, attenuation_np_mm: float, depth_mm: float) -> float:
    """Compute attenuated amplitude after round-trip propagation."""
    return initial * np.exp(-attenuation_np_mm * 2.0 * depth_mm)


def generate_trace(params: TraceParams) -> np.ndarray:
    """Generate a clean pulse-echo A-scan trace (no noise applied)."""
    dt_us = 1.0 / params.sampling_rate_mhz
    t = np.arange(params.n_samples) * dt_us

    t_surface = SURFACE_ECHO_OFFSET_US
    a_surface = 1.0
    signal = a_surface * generate_pulse(t, params.center_freq_mhz, params.pulse_sigma_us, t_surface)

    t_backwall = t_surface + compute_arrival_time(params.thickness_mm, params.velocity_ms)
    a_backwall = compute_amplitude(a_surface, params.attenuation_np_mm, params.thickness_mm)
    signal += a_backwall * generate_pulse(
        t, params.center_freq_mhz, params.pulse_sigma_us, t_backwall
    )

    if params.has_defect:
        if params.defect_depth_mm >= params.thickness_mm:
            raise ValueError(
                f"defect_depth_mm ({params.defect_depth_mm}) must be < "
                f"thickness_mm ({params.thickness_mm})"
            )
        t_defect = t_surface + compute_arrival_time(params.defect_depth_mm, params.velocity_ms)
        a_defect = compute_amplitude(
            params.defect_reflectivity, params.attenuation_np_mm, params.defect_depth_mm
        )
        signal += a_defect * generate_pulse(
            t, params.center_freq_mhz, params.pulse_sigma_us, t_defect
        )

    return signal
