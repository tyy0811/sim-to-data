"""Regime configuration for source, shifted, and randomized parameter ranges."""

from dataclasses import dataclass
import numpy as np
import yaml

from simtodata.simulator.defects import sample_defect
from simtodata.simulator.forward_model import TraceParams


@dataclass
class RegimeConfig:
    """Parameter ranges for a simulation regime."""

    name: str
    thickness_mm: tuple
    velocity_ms: tuple
    attenuation_np_mm: tuple
    center_freq_mhz: tuple
    pulse_sigma_us: tuple
    defect_depth_mm: tuple
    snr_db: tuple
    baseline_drift: tuple
    gain_variation: tuple
    jitter_samples: tuple
    dropout_n_gaps: tuple
    dropout_gap_length: tuple


def _as_tuple(val):
    if isinstance(val, list):
        return tuple(val)
    return (val, val)


def load_regimes_from_yaml(yaml_path: str) -> dict:
    """Load all regime configurations from a YAML file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    regimes = {}
    for regime_name in ["source_regime", "shifted_regime", "randomized_regime"]:
        rc = config[regime_name]
        dropout = rc.get("masked_dropout", {})
        regimes[regime_name.replace("_regime", "")] = RegimeConfig(
            name=regime_name.replace("_regime", ""),
            thickness_mm=_as_tuple(rc["material_thickness_mm"]),
            velocity_ms=_as_tuple(rc["material_velocity_ms"]),
            attenuation_np_mm=_as_tuple(rc["attenuation_np_mm"]),
            center_freq_mhz=_as_tuple(rc["center_freq_mhz"]),
            pulse_sigma_us=_as_tuple(rc["pulse_sigma_us"]),
            defect_depth_mm=_as_tuple(rc["defect_depth_mm"]),
            snr_db=_as_tuple(rc["snr_db"]),
            baseline_drift=_as_tuple(rc.get("baseline_drift", 0)),
            gain_variation=_as_tuple(rc.get("gain_variation", [1.0, 1.0])),
            jitter_samples=_as_tuple(rc.get("jitter_samples", 0)),
            dropout_n_gaps=_as_tuple(dropout.get("n_gaps", 0)),
            dropout_gap_length=_as_tuple(dropout.get("gap_length", [0, 0])),
        )
    return regimes


def sample_trace_params(
    regime: RegimeConfig,
    rng: np.random.Generator,
    severity_class: int,
    n_samples: int = 1024,
    sampling_rate_mhz: float = 50.0,
) -> TraceParams:
    """Sample trace parameters from a regime configuration."""
    thickness = rng.uniform(*regime.thickness_mm)
    defect = sample_defect(
        rng, severity_class, regime.defect_depth_mm, thickness_mm=thickness
    )

    jitter = (
        int(rng.integers(regime.jitter_samples[0], regime.jitter_samples[1] + 1))
        if regime.jitter_samples[1] > 0
        else 0
    )
    n_gaps = (
        int(rng.integers(regime.dropout_n_gaps[0], regime.dropout_n_gaps[1] + 1))
        if regime.dropout_n_gaps[1] > 0
        else 0
    )

    return TraceParams(
        thickness_mm=thickness,
        velocity_ms=rng.uniform(*regime.velocity_ms),
        attenuation_np_mm=rng.uniform(*regime.attenuation_np_mm),
        center_freq_mhz=rng.uniform(*regime.center_freq_mhz),
        pulse_sigma_us=rng.uniform(*regime.pulse_sigma_us),
        has_defect=severity_class > 0,
        defect_depth_mm=defect.depth_mm,
        defect_reflectivity=defect.reflectivity,
        severity_label=severity_class,
        snr_db=rng.uniform(*regime.snr_db),
        baseline_drift_amplitude=rng.uniform(*regime.baseline_drift),
        gain_variation=rng.uniform(*regime.gain_variation),
        jitter_samples=jitter,
        n_dropout_gaps=n_gaps,
        dropout_gap_length=regime.dropout_gap_length,
        n_samples=n_samples,
        sampling_rate_mhz=sampling_rate_mhz,
    )
