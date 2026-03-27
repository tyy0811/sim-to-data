# sim-to-data Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a synthetic ultrasonic inspection pipeline that trains and evaluates defect detectors under sensor and material domain shift.

**Architecture:** A pulse-echo forward model generates 1D A-scan traces with configurable defects and noise. Three regimes (source/shifted/randomized) create controlled domain shift. Non-neural baselines and a 1D CNN are benchmarked across transfer, adaptation, and robustness experiments.

**Tech Stack:** Python 3.10+, PyTorch (CPU), NumPy, SciPy, scikit-learn, matplotlib, PyYAML, pytest, ruff

**Design doc:** `docs/plans/2026-03-27-sim-to-data-design.md`

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/simtodata/__init__.py`, all subpackage `__init__.py` files

**Step 1: Initialize git repo**

```bash
cd /Users/zenith/Desktop/sim-to-data
git init
```

**Step 2: Create `.gitignore`**

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
data/
models/
results/
docs/figures/*.png
*.npz
*.pt
*.joblib
.ruff_cache/
.pytest_cache/
```

**Step 3: Create `pyproject.toml`**

```toml
[project]
name = "simtodata"
version = "0.1.0"
description = "Synthetic ultrasonic inspection pipeline for defect detection under domain shift"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "torch>=2.0",
    "pyyaml>=6.0",
    "scikit-learn>=1.2",
    "matplotlib>=3.7",
    "joblib>=1.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Create directory structure and empty `__init__.py` files**

```bash
mkdir -p src/simtodata/simulator src/simtodata/data src/simtodata/features \
         src/simtodata/models src/simtodata/evaluation \
         configs experiments tests docs/figures docs/plans
touch src/simtodata/__init__.py src/simtodata/simulator/__init__.py \
      src/simtodata/data/__init__.py src/simtodata/features/__init__.py \
      src/simtodata/models/__init__.py src/simtodata/evaluation/__init__.py
```

**Step 5: Install in editable mode and verify**

```bash
pip install -e ".[dev]"
python -c "import simtodata; print('OK')"
```

**Step 6: Commit**

```bash
git add .gitignore pyproject.toml src/
git commit -m "chore: project scaffold with package structure"
```

---

## Task 2: Forward Model and Defects

**Files:**
- Create: `src/simtodata/simulator/forward_model.py`
- Create: `src/simtodata/simulator/defects.py`
- Create: `tests/test_forward_model.py`

**Step 1: Write tests**

`tests/test_forward_model.py`:
```python
"""Tests for pulse-echo forward model and defect classification."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from simtodata.simulator.forward_model import (
    SURFACE_ECHO_OFFSET_US,
    TraceParams,
    compute_amplitude,
    compute_arrival_time,
    generate_pulse,
    generate_trace,
)
from simtodata.simulator.defects import (
    DefectConfig,
    HIGH_REFLECTIVITY_RANGE,
    LOW_REFLECTIVITY_RANGE,
    classify_severity,
    sample_defect,
)


def _default_params(**kwargs):
    defaults = dict(
        thickness_mm=20.0,
        velocity_ms=6000.0,
        attenuation_np_mm=0.02,
        center_freq_mhz=3.0,
        pulse_sigma_us=1.0,
    )
    defaults.update(kwargs)
    return TraceParams(**defaults)


class TestGeneratePulse:
    def test_shape(self):
        t = np.linspace(0, 20, 1024)
        pulse = generate_pulse(t, 3.0, 1.0, 5.0)
        assert pulse.shape == (1024,)

    def test_finite(self):
        t = np.linspace(0, 20, 1024)
        pulse = generate_pulse(t, 3.0, 1.0, 5.0)
        assert np.all(np.isfinite(pulse))

    def test_peak_near_center(self):
        t = np.linspace(0, 20, 1024)
        pulse = generate_pulse(t, 3.0, 1.0, 5.0)
        peak_idx = np.argmax(np.abs(pulse))
        peak_time = t[peak_idx]
        assert abs(peak_time - 5.0) < 1.0


class TestComputeArrivalTime:
    def test_basic(self):
        # 20mm, 6000 m/s = 6 mm/us, round-trip 40mm => 6.667 us
        t = compute_arrival_time(20.0, 6000.0)
        assert abs(t - 6.667) < 0.01

    def test_zero_depth(self):
        assert compute_arrival_time(0.0, 6000.0) == 0.0


class TestComputeAmplitude:
    def test_no_attenuation(self):
        assert compute_amplitude(1.0, 0.0, 20.0) == 1.0

    def test_with_attenuation(self):
        a = compute_amplitude(1.0, 0.02, 20.0)
        expected = np.exp(-0.02 * 2 * 20.0)
        assert abs(a - expected) < 1e-10


class TestGenerateTrace:
    def test_shape(self):
        signal = generate_trace(_default_params())
        assert signal.shape == (1024,)

    def test_finite(self):
        signal = generate_trace(_default_params())
        assert np.all(np.isfinite(signal))

    def test_backwall_after_surface(self):
        signal = generate_trace(_default_params())
        abs_signal = np.abs(signal)
        peaks, props = find_peaks(abs_signal, height=0.1 * np.max(abs_signal))
        sorted_peaks = peaks[np.argsort(props["peak_heights"])[::-1]]
        top2 = sorted(sorted_peaks[:2])
        assert len(top2) == 2
        assert top2[0] < top2[1]

    def test_defect_between_surface_and_backwall(self):
        params = _default_params(
            has_defect=True, defect_depth_mm=10.0, defect_reflectivity=0.5
        )
        signal = generate_trace(params)
        abs_signal = np.abs(signal)
        peaks, props = find_peaks(abs_signal, height=0.05 * np.max(abs_signal))
        sorted_peaks = peaks[np.argsort(props["peak_heights"])[::-1]]
        top3 = sorted(sorted_peaks[:3])
        assert len(top3) >= 3
        assert top3[0] < top3[1] < top3[2]

    def test_higher_attenuation_reduces_backwall(self):
        sig_low = generate_trace(_default_params(attenuation_np_mm=0.01))
        sig_high = generate_trace(_default_params(attenuation_np_mm=0.08))
        t_bw = SURFACE_ECHO_OFFSET_US + compute_arrival_time(20.0, 6000.0)
        bw_idx = int(t_bw * 50.0)
        w = slice(max(0, bw_idx - 50), min(1024, bw_idx + 50))
        assert np.max(np.abs(sig_low[w])) > np.max(np.abs(sig_high[w]))

    def test_no_defect_nonzero(self):
        signal = generate_trace(_default_params(has_defect=False))
        assert np.max(np.abs(signal)) > 0

    def test_deterministic(self):
        params = _default_params(has_defect=True, defect_depth_mm=10.0, defect_reflectivity=0.5)
        np.testing.assert_array_equal(generate_trace(params), generate_trace(params))


class TestClassifySeverity:
    def test_no_defect(self):
        assert classify_severity(0.0) == 0

    def test_low_severity(self):
        assert classify_severity(0.2) == 1

    def test_high_severity(self):
        assert classify_severity(0.5) == 2

    def test_boundary(self):
        assert classify_severity(0.3) == 1
        assert classify_severity(0.4) == 2


class TestSampleDefect:
    def test_no_defect(self):
        rng = np.random.default_rng(42)
        d = sample_defect(rng, 0, (2.0, 28.0))
        assert d.severity_label == 0
        assert d.reflectivity == 0.0

    def test_low_severity_range(self):
        rng = np.random.default_rng(42)
        d = sample_defect(rng, 1, (2.0, 28.0))
        assert d.severity_label == 1
        assert LOW_REFLECTIVITY_RANGE[0] <= d.reflectivity <= LOW_REFLECTIVITY_RANGE[1]

    def test_high_severity_range(self):
        rng = np.random.default_rng(42)
        d = sample_defect(rng, 2, (2.0, 28.0))
        assert d.severity_label == 2
        assert HIGH_REFLECTIVITY_RANGE[0] <= d.reflectivity <= HIGH_REFLECTIVITY_RANGE[1]

    def test_depth_in_range(self):
        rng = np.random.default_rng(42)
        d = sample_defect(rng, 1, (5.0, 15.0))
        assert 5.0 <= d.depth_mm <= 15.0
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_forward_model.py -v
```

Expected: FAIL (modules not found)

**Step 3: Implement `forward_model.py`**

`src/simtodata/simulator/forward_model.py`:
```python
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
        t_defect = t_surface + compute_arrival_time(params.defect_depth_mm, params.velocity_ms)
        a_defect = compute_amplitude(
            params.defect_reflectivity, params.attenuation_np_mm, params.defect_depth_mm
        )
        signal += a_defect * generate_pulse(
            t, params.center_freq_mhz, params.pulse_sigma_us, t_defect
        )

    return signal
```

**Step 4: Implement `defects.py`**

`src/simtodata/simulator/defects.py`:
```python
"""Defect parameterization and severity classification."""

from dataclasses import dataclass
import numpy as np

LOW_REFLECTIVITY_RANGE = (0.1, 0.3)
HIGH_REFLECTIVITY_RANGE = (0.4, 0.8)


@dataclass
class DefectConfig:
    """Configuration for a single defect."""

    depth_mm: float
    reflectivity: float
    severity_label: int


def classify_severity(reflectivity: float) -> int:
    """Classify defect severity: 0=none, 1=low, 2=high."""
    if reflectivity <= 0:
        return 0
    if reflectivity <= LOW_REFLECTIVITY_RANGE[1]:
        return 1
    return 2


def sample_defect(rng: np.random.Generator, severity: int, depth_range: tuple) -> DefectConfig:
    """Sample a defect configuration for a given severity class."""
    if severity == 0:
        return DefectConfig(depth_mm=0.0, reflectivity=0.0, severity_label=0)
    depth = rng.uniform(depth_range[0], depth_range[1])
    if severity == 1:
        reflectivity = rng.uniform(*LOW_REFLECTIVITY_RANGE)
    else:
        reflectivity = rng.uniform(*HIGH_REFLECTIVITY_RANGE)
    return DefectConfig(depth_mm=depth, reflectivity=reflectivity, severity_label=severity)
```

**Step 5: Run tests — verify they pass**

```bash
pytest tests/test_forward_model.py -v
```

Expected: all 19 tests PASS

**Step 6: Commit**

```bash
git add src/simtodata/simulator/forward_model.py src/simtodata/simulator/defects.py tests/test_forward_model.py
git commit -m "feat: pulse-echo forward model and defect severity classification"
```

---

## Task 3: Noise Injection

**Files:**
- Create: `src/simtodata/simulator/noise.py`
- Create: `tests/test_noise.py`

**Step 1: Write tests**

`tests/test_noise.py`:
```python
"""Tests for noise injection functions."""

import numpy as np
import pytest

from simtodata.simulator.noise import (
    add_baseline_drift,
    add_gain_variation,
    add_gaussian_noise,
    add_masked_dropout,
    add_temporal_jitter,
    apply_all_noise,
)


@pytest.fixture
def clean_signal():
    t = np.linspace(0, 1, 1024)
    return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


class TestGaussianNoise:
    def test_changes_signal(self, clean_signal, rng):
        noisy = add_gaussian_noise(clean_signal, 20.0, rng)
        assert not np.allclose(noisy, clean_signal)

    def test_deterministic(self, clean_signal):
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        np.testing.assert_array_equal(
            add_gaussian_noise(clean_signal, 20.0, r1),
            add_gaussian_noise(clean_signal, 20.0, r2),
        )

    def test_zero_signal_unchanged(self, rng):
        signal = np.zeros(1024)
        result = add_gaussian_noise(signal, 20.0, rng)
        np.testing.assert_array_equal(result, signal)


class TestBaselineDrift:
    def test_changes_signal(self, clean_signal, rng):
        assert not np.allclose(add_baseline_drift(clean_signal, 0.2, rng), clean_signal)

    def test_zero_amplitude_unchanged(self, clean_signal, rng):
        np.testing.assert_array_equal(add_baseline_drift(clean_signal, 0.0, rng), clean_signal)

    def test_deterministic(self, clean_signal):
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        np.testing.assert_array_equal(
            add_baseline_drift(clean_signal, 0.2, r1),
            add_baseline_drift(clean_signal, 0.2, r2),
        )


class TestTemporalJitter:
    def test_zero_shift_unchanged(self, clean_signal, rng):
        np.testing.assert_array_equal(add_temporal_jitter(clean_signal, 0, rng), clean_signal)

    def test_no_circular_wrap(self):
        signal = np.arange(1.0, 101.0)
        rng = np.random.default_rng(42)
        jittered = add_temporal_jitter(signal, 5, rng)
        if np.allclose(jittered, signal):
            return  # shift was 0
        if jittered[0] == 0.0:  # positive shift: zero-filled left
            first_nz = np.argmax(jittered != 0)
            np.testing.assert_array_equal(jittered[first_nz:], signal[: len(signal) - first_nz])
        else:  # negative shift: zero-filled right
            last_nz = len(jittered) - 1 - np.argmax(jittered[::-1] != 0)
            assert np.all(jittered[last_nz + 1 :] == 0.0)

    def test_deterministic(self, clean_signal):
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        np.testing.assert_array_equal(
            add_temporal_jitter(clean_signal, 2, r1),
            add_temporal_jitter(clean_signal, 2, r2),
        )


class TestMaskedDropout:
    def test_zero_gaps_unchanged(self, clean_signal, rng):
        np.testing.assert_array_equal(
            add_masked_dropout(clean_signal, 0, (5, 10), rng), clean_signal
        )

    def test_creates_contiguous_zeros(self, clean_signal, rng):
        result = add_masked_dropout(clean_signal, 2, (10, 20), rng)
        zeroed = np.where(result == 0.0)[0]
        assert len(zeroed) > 0
        # Check contiguity: within each gap, indices are consecutive
        gaps = np.split(zeroed, np.where(np.diff(zeroed) > 1)[0] + 1)
        for gap in gaps:
            if len(gap) > 1:
                assert np.all(np.diff(gap) == 1)

    def test_deterministic(self, clean_signal):
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        np.testing.assert_array_equal(
            add_masked_dropout(clean_signal, 2, (10, 20), r1),
            add_masked_dropout(clean_signal, 2, (10, 20), r2),
        )


class TestGainVariation:
    def test_unity_unchanged(self, clean_signal):
        np.testing.assert_array_equal(add_gain_variation(clean_signal, 1.0), clean_signal)

    def test_scales(self, clean_signal):
        np.testing.assert_allclose(add_gain_variation(clean_signal, 2.0), clean_signal * 2.0)


class TestApplyAllNoise:
    def _noise_params(self):
        """Create a simple namespace with noise parameters for testing."""
        from types import SimpleNamespace
        return SimpleNamespace(
            snr_db=20.0, baseline_drift_amplitude=0.1, gain_variation=1.1,
            jitter_samples=1, n_dropout_gaps=1, dropout_gap_length=(5, 10),
        )

    def test_deterministic(self, clean_signal):
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        params = self._noise_params()
        np.testing.assert_array_equal(
            apply_all_noise(clean_signal, params, r1),
            apply_all_noise(clean_signal, params, r2),
        )

    def test_changes_signal(self, clean_signal, rng):
        params = self._noise_params()
        result = apply_all_noise(clean_signal, params, rng)
        assert not np.allclose(result, clean_signal)
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_noise.py -v
```

**Step 3: Implement**

`src/simtodata/simulator/noise.py`:
```python
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
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_noise.py -v
```

Expected: all 16 tests PASS

**Step 5: Commit**

```bash
git add src/simtodata/simulator/noise.py tests/test_noise.py
git commit -m "feat: noise injection with zero-padded jitter and contiguous dropout"
```

---

## Task 4: Regime Configuration

**Files:**
- Create: `src/simtodata/simulator/regime.py`
- Create: `configs/simulator.yaml`
- Create: `tests/test_regime.py`

**Step 1: Write `configs/simulator.yaml`**

```yaml
signal:
  n_samples: 1024
  sampling_rate_mhz: 50.0

source_regime:
  material_thickness_mm: [10.0, 30.0]
  material_velocity_ms: [5800.0, 6200.0]
  attenuation_np_mm: [0.01, 0.05]
  center_freq_mhz: [2.0, 5.0]
  pulse_sigma_us: [0.5, 1.5]
  defect_depth_mm: [2.0, 28.0]
  defect_reflectivity: [0.1, 0.8]
  snr_db: [20.0, 40.0]
  baseline_drift: 0.0
  gain_variation: [1.0, 1.0]
  jitter_samples: 0
  masked_dropout:
    n_gaps: 0
    gap_length: [0, 0]

shifted_regime:
  material_thickness_mm: [10.0, 30.0]
  material_velocity_ms: [5500.0, 6500.0]
  attenuation_np_mm: [0.01, 0.10]
  center_freq_mhz: [1.5, 7.0]
  pulse_sigma_us: [0.3, 2.0]
  defect_depth_mm: [2.0, 28.0]
  defect_reflectivity: [0.05, 0.9]
  snr_db: [5.0, 40.0]
  baseline_drift: [0.0, 0.3]
  gain_variation: [0.7, 1.3]
  jitter_samples: [0, 2]
  masked_dropout:
    n_gaps: [0, 3]
    gap_length: [5, 20]

# Domain randomization: same ranges as shifted.
# The difference is that source training never touches these ranges.
# This is a tunable choice — if B3 degrades source performance vs B1,
# consider 50/50 mixture of source/shifted sampling.
randomized_regime:
  material_thickness_mm: [10.0, 30.0]
  material_velocity_ms: [5500.0, 6500.0]
  attenuation_np_mm: [0.01, 0.10]
  center_freq_mhz: [1.5, 7.0]
  pulse_sigma_us: [0.3, 2.0]
  defect_depth_mm: [2.0, 28.0]
  defect_reflectivity: [0.05, 0.9]
  snr_db: [5.0, 40.0]
  baseline_drift: [0.0, 0.3]
  gain_variation: [0.7, 1.3]
  jitter_samples: [0, 2]
  masked_dropout:
    n_gaps: [0, 3]
    gap_length: [5, 20]

dataset_sizes:
  train: 20000
  val: 2000
  test_source: 3000
  test_shifted: 3000
  adapt: 200
  train_randomized: 20000

class_distribution:
  no_defect: 0.33
  low_severity: 0.33
  high_severity: 0.34

severity_thresholds:
  low_reflectivity: [0.1, 0.3]
  high_reflectivity: [0.4, 0.8]

seed: 42
```

**Step 2: Write tests**

`tests/test_regime.py`:
```python
"""Tests for regime configuration and parameter sampling."""

import numpy as np
import pytest
import yaml

from simtodata.simulator.regime import RegimeConfig, load_regimes_from_yaml, sample_trace_params


def _source_regime():
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


class TestRegimeConfig:
    def test_sample_params_in_range(self):
        regime = _source_regime()
        rng = np.random.default_rng(42)
        params = sample_trace_params(regime, rng, severity_class=1)
        assert 5800 <= params.velocity_ms <= 6200
        assert 0.01 <= params.attenuation_np_mm <= 0.05
        assert 2.0 <= params.center_freq_mhz <= 5.0
        assert 20.0 <= params.snr_db <= 40.0

    def test_deterministic(self):
        regime = _source_regime()
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        p1 = sample_trace_params(regime, r1, severity_class=1)
        p2 = sample_trace_params(regime, r2, severity_class=1)
        assert p1.velocity_ms == p2.velocity_ms
        assert p1.defect_depth_mm == p2.defect_depth_mm

    def test_no_defect_class(self):
        regime = _source_regime()
        rng = np.random.default_rng(42)
        params = sample_trace_params(regime, rng, severity_class=0)
        assert params.has_defect is False
        assert params.severity_label == 0

    def test_defect_depth_within_thickness(self):
        regime = _source_regime()
        rng = np.random.default_rng(42)
        for _ in range(100):
            params = sample_trace_params(regime, rng, severity_class=2)
            assert params.defect_depth_mm < params.thickness_mm


class TestLoadRegimes:
    def test_load_from_yaml(self):
        regimes = load_regimes_from_yaml("configs/simulator.yaml")
        assert "source" in regimes
        assert "shifted" in regimes
        assert "randomized" in regimes

    def test_shifted_wider_than_source(self):
        regimes = load_regimes_from_yaml("configs/simulator.yaml")
        assert regimes["shifted"].snr_db[0] < regimes["source"].snr_db[0]
        assert regimes["shifted"].velocity_ms[1] > regimes["source"].velocity_ms[1]
```

**Step 3: Implement**

`src/simtodata/simulator/regime.py`:
```python
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
    max_defect_depth = min(regime.defect_depth_mm[1], thickness - 1.0)
    defect = sample_defect(rng, severity_class, (regime.defect_depth_mm[0], max_defect_depth))

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
```

**Step 4: Run tests**

```bash
pytest tests/test_regime.py -v
```

Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add src/simtodata/simulator/regime.py configs/simulator.yaml tests/test_regime.py
git commit -m "feat: regime configuration with source, shifted, and randomized ranges"
```

---

## Task 5: Data Generation

**Files:**
- Create: `src/simtodata/data/generate.py`
- Create: `tests/test_dataset.py`

**Step 1: Write tests**

`tests/test_dataset.py`:
```python
"""Tests for dataset generation, loading, and transforms."""

import numpy as np
import os
import pytest
import tempfile

from simtodata.data.generate import generate_dataset
from simtodata.simulator.regime import RegimeConfig


def _source_regime():
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


class TestGenerateDataset:
    def test_shapes(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 100, seed=42, class_distribution=class_dist)
        assert data["signals"].shape == (100, 1024)
        assert data["labels"].shape == (100,)

    def test_labels_valid(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 300, seed=42, class_distribution=class_dist)
        assert set(np.unique(data["labels"])) == {0, 1, 2}

    def test_approximate_class_balance(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 3000, seed=42, class_distribution=class_dist)
        counts = np.bincount(data["labels"])
        for c in counts:
            assert abs(c / 3000 - 0.33) < 0.05

    def test_deterministic(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        d1 = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        d2 = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        np.testing.assert_array_equal(d1["signals"], d2["signals"])
        np.testing.assert_array_equal(d1["labels"], d2["labels"])

    def test_signals_finite(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        assert np.all(np.isfinite(data["signals"]))

    def test_save_and_load(self):
        regime = _source_regime()
        class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
        data = generate_dataset(regime, 50, seed=42, class_distribution=class_dist)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            np.savez(path, signals=data["signals"], labels=data["labels"])
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded["signals"], data["signals"])
            np.testing.assert_array_equal(loaded["labels"], data["labels"])
```

**Step 2: Implement**

`src/simtodata/data/generate.py`:
```python
"""Dataset generation orchestrator."""

import argparse
import os
import time

import numpy as np
import yaml

from simtodata.simulator.forward_model import generate_trace
from simtodata.simulator.noise import apply_all_noise
from simtodata.simulator.regime import load_regimes_from_yaml, sample_trace_params


def generate_dataset(regime, n_samples, seed, class_distribution, n_signal_samples=1024,
                     sampling_rate_mhz=50.0):
    """Generate a dataset of synthetic A-scan traces.

    Args:
        regime: RegimeConfig with parameter ranges.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        class_distribution: Dict with keys 'no_defect', 'low_severity', 'high_severity'.
        n_signal_samples: Samples per trace.
        sampling_rate_mhz: Sampling rate.

    Returns:
        Dict with 'signals' (n, 1024) and 'labels' (n,).
    """
    rng = np.random.default_rng(seed)
    signals = np.zeros((n_samples, n_signal_samples))
    labels = np.zeros(n_samples, dtype=np.int64)

    classes = [0, 1, 2]
    probs = [
        class_distribution["no_defect"],
        class_distribution["low_severity"],
        class_distribution["high_severity"],
    ]

    for i in range(n_samples):
        severity = rng.choice(classes, p=probs)
        params = sample_trace_params(regime, rng, severity, n_signal_samples, sampling_rate_mhz)
        clean = generate_trace(params)
        noisy = apply_all_noise(clean, params, rng)
        signals[i] = noisy
        labels[i] = severity

    return {"signals": signals, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic inspection datasets")
    parser.add_argument("--config", default="configs/simulator.yaml")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--quick", action="store_true", help="Small datasets for testing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    regimes = load_regimes_from_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    sizes = config["dataset_sizes"]
    if args.quick:
        sizes = {k: min(v, 100) for k, v in sizes.items()}

    class_dist = config["class_distribution"]
    rng = np.random.default_rng(config["seed"])

    datasets = [
        ("source_train", "source", sizes["train"]),
        ("source_val", "source", sizes["val"]),
        ("source_test", "source", sizes["test_source"]),
        ("shifted_test", "shifted", sizes["test_shifted"]),
        ("shifted_adapt", "shifted", sizes["adapt"]),
        ("randomized_train", "randomized", sizes["train_randomized"]),
    ]

    for name, regime_name, n in datasets:
        print(f"Generating {name} ({n} samples)...")
        t0 = time.time()
        split_seed = int(rng.integers(0, 2**31))
        data = generate_dataset(regimes[regime_name], n, split_seed, class_dist)
        path = os.path.join(args.output_dir, f"{name}.npz")
        np.savez(path, signals=data["signals"], labels=data["labels"])
        print(f"  Saved to {path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
```

**Step 3: Run tests**

```bash
pytest tests/test_dataset.py -v
```

Expected: all 6 tests PASS

**Step 4: Commit**

```bash
git add src/simtodata/data/generate.py tests/test_dataset.py
git commit -m "feat: dataset generation pipeline with class-balanced sampling"
```

---

## Task 6: PyTorch Dataset and Transforms

**Files:**
- Create: `src/simtodata/data/dataset.py`
- Create: `src/simtodata/data/transforms.py`

**Step 1: Implement**

`src/simtodata/data/transforms.py`:
```python
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
```

`src/simtodata/data/dataset.py`:
```python
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
```

**Step 2: Add tests to `tests/test_dataset.py`** (append to existing file)

Add these tests to the existing `tests/test_dataset.py`:
```python
import torch
from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize


class TestInspectionDataset:
    def test_tensor_shapes(self, tmp_path):
        signals = np.random.randn(50, 1024).astype(np.float32)
        labels = np.array([0] * 17 + [1] * 17 + [2] * 16, dtype=np.int64)
        np.savez(tmp_path / "test.npz", signals=signals, labels=labels)
        ds = InspectionDataset(str(tmp_path / "test.npz"))
        signal, label = ds[0]
        assert signal.shape == (1, 1024)
        assert label.shape == ()
        assert signal.dtype == torch.float32

    def test_length(self, tmp_path):
        signals = np.random.randn(50, 1024).astype(np.float32)
        labels = np.zeros(50, dtype=np.int64)
        np.savez(tmp_path / "test.npz", signals=signals, labels=labels)
        ds = InspectionDataset(str(tmp_path / "test.npz"))
        assert len(ds) == 50

    def test_normalize_transform(self, tmp_path):
        signals = np.random.randn(10, 1024).astype(np.float32) * 5 + 3
        labels = np.zeros(10, dtype=np.int64)
        np.savez(tmp_path / "test.npz", signals=signals, labels=labels)
        ds = InspectionDataset(str(tmp_path / "test.npz"), transform=Normalize())
        signal, _ = ds[0]
        assert abs(signal.mean().item()) < 0.01
        assert abs(signal.std().item() - 1.0) < 0.01
```

**Step 3: Run tests**

```bash
pytest tests/test_dataset.py -v
```

Expected: all 9 tests PASS

**Step 4: Commit**

```bash
git add src/simtodata/data/dataset.py src/simtodata/data/transforms.py tests/test_dataset.py
git commit -m "feat: PyTorch dataset class with normalization transform"
```

---

## Task 7: 1D CNN Model and Factory

**Files:**
- Create: `src/simtodata/models/cnn1d.py`
- Create: `src/simtodata/models/factory.py`
- Create: `configs/model_cnn1d.yaml`
- Create: `tests/test_model_cnn1d.py`

**Step 1: Write tests**

`tests/test_model_cnn1d.py`:
```python
"""Tests for the 1D CNN defect classifier."""

import torch
import pytest

from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.factory import model_from_config


class TestDefectCNN1D:
    def test_output_shape(self):
        model = DefectCNN1D()
        x = torch.randn(16, 1, 1024)
        out = model(x)
        assert out.shape == (16, 3)

    def test_single_sample(self):
        model = DefectCNN1D()
        x = torch.randn(1, 1, 1024)
        out = model(x)
        assert out.shape == (1, 3)

    def test_large_batch(self):
        model = DefectCNN1D()
        x = torch.randn(256, 1, 1024)
        out = model(x)
        assert out.shape == (256, 3)

    def test_gradients_flow(self):
        model = DefectCNN1D()
        x = torch.randn(4, 1, 1024)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_param_count(self):
        model = DefectCNN1D()
        count = model.param_count()
        assert 20_000 < count < 200_000


class TestFactory:
    def test_from_config(self):
        model = model_from_config("configs/model_cnn1d.yaml")
        assert isinstance(model, DefectCNN1D)
        x = torch.randn(2, 1, 1024)
        out = model(x)
        assert out.shape == (2, 3)
```

**Step 2: Create config**

`configs/model_cnn1d.yaml`:
```yaml
architecture:
  type: cnn_1d
  channels: [32, 64, 128]
  kernels: [7, 5, 3]
  fc_hidden: 64
  dropout: 0.3
  num_classes: 3

training:
  epochs: 50
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler_patience: 5
  scheduler_factor: 0.5
  early_stopping_patience: 10

finetune:
  epochs: 20
  learning_rate: 0.0001
```

**Step 3: Implement**

`src/simtodata/models/cnn1d.py`:
```python
"""1D CNN classifier for defect severity detection."""

import torch.nn as nn


class DefectCNN1D(nn.Module):
    """1D convolutional network for A-scan classification."""

    def __init__(self, channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels[0], kernels[0], stride=2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[0], channels[1], kernels[1], stride=1),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[1], channels[2], kernels[2], stride=1),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels[2], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
```

`src/simtodata/models/factory.py`:
```python
"""Model factory: config -> nn.Module."""

import yaml

from simtodata.models.cnn1d import DefectCNN1D


def model_from_config(config_path: str):
    """Instantiate a model from a YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    arch = config["architecture"]
    if arch["type"] == "cnn_1d":
        return DefectCNN1D(
            channels=tuple(arch["channels"]),
            kernels=tuple(arch["kernels"]),
            fc_hidden=arch["fc_hidden"],
            dropout=arch["dropout"],
            num_classes=arch["num_classes"],
        )
    raise ValueError(f"Unknown architecture type: {arch['type']}")
```

**Step 4: Run tests**

```bash
pytest tests/test_model_cnn1d.py -v
```

Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add src/simtodata/models/cnn1d.py src/simtodata/models/factory.py \
        configs/model_cnn1d.yaml tests/test_model_cnn1d.py
git commit -m "feat: 1D CNN classifier with config-driven factory"
```

---

## Task 8: Training Loop and Prediction

**Files:**
- Create: `src/simtodata/models/train.py`
- Create: `src/simtodata/models/predict.py`

**Step 1: Implement `train.py`**

`src/simtodata/models/train.py`:
```python
"""Model training with early stopping and LR scheduling."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def _evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    n = 0
    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            n += len(labels)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / n
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3,
                weight_decay=1e-4, patience=10, scheduler_patience=5,
                scheduler_factor=0.5, device="cpu"):
    """Train a model with optional early stopping and LR scheduling.

    Args:
        model: nn.Module to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (None to skip validation/early stopping).
        epochs: Maximum training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience (epochs without val improvement).
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR reduction factor.
        device: Device string.

    Returns:
        (model, history) where history is a dict of per-epoch metrics.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=scheduler_patience, factor=scheduler_factor
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)

        train_loss = total_loss / n
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, val_f1 = _evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_f1: {val_f1:.4f}"
                )
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
```

**Step 2: Implement `predict.py`**

`src/simtodata/models/predict.py`:
```python
"""Model inference and probability output."""

import numpy as np
import torch


def predict_batch(model, dataloader, device="cpu"):
    """Run inference on a DataLoader.

    Returns:
        (predictions, probabilities, labels) as numpy arrays.
    """
    model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs), np.concatenate(all_labels)
```

**Step 3: Commit**

```bash
git add src/simtodata/models/train.py src/simtodata/models/predict.py
git commit -m "feat: training loop with early stopping and batch prediction"
```

---

## Task 9: Metrics Suite

> **Reordered:** Metrics must exist before any experiment script can call `compute_all_metrics`.

**Files:**
- Create: `src/simtodata/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write tests**

`tests/test_metrics.py`:
```python
"""Tests for evaluation metrics."""

import numpy as np
import pytest

from simtodata.evaluation.metrics import (
    compute_all_metrics,
    compute_auroc,
    compute_ece,
    compute_macro_f1,
    compute_per_class_metrics,
)


class TestMacroF1:
    def test_perfect(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert compute_macro_f1(y_true, y_pred) == 1.0

    def test_random(self):
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1, 2], size=1000)
        y_pred = rng.choice([0, 1, 2], size=1000)
        f1 = compute_macro_f1(y_true, y_pred)
        assert 0.1 < f1 < 0.6


class TestAUROC:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
        ])
        assert compute_auroc(y_true, y_proba) > 0.99


class TestECE:
    def test_perfect_calibration(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        ece = compute_ece(y_true, y_proba)
        assert ece < 0.01

    def test_overconfident(self):
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_proba = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
            [0.99, 0.005, 0.005],  # wrong but overconfident
            [0.005, 0.99, 0.005],  # wrong but overconfident
            [0.005, 0.005, 0.99],
        ])
        ece = compute_ece(y_true, y_proba)
        assert ece > 0

    def test_no_division_by_zero(self):
        y_true = np.array([0, 0, 0])
        y_proba = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        ece = compute_ece(y_true, y_proba)
        assert np.isfinite(ece)


class TestPerClassMetrics:
    def test_structure(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        result = compute_per_class_metrics(y_true, y_pred)
        assert "precision" in result
        assert "recall" in result
        assert len(result["precision"]) == 3


class TestComputeAllMetrics:
    def test_all_keys_present(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        assert "macro_f1" in metrics
        assert "auroc" in metrics
        assert "ece" in metrics
        assert "per_class" in metrics
```

**Step 2: Implement**

`src/simtodata/evaluation/metrics.py`:
```python
"""Evaluation metrics: F1, AUROC, ECE, per-class precision/recall."""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score


def compute_macro_f1(y_true, y_pred):
    """Compute macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_auroc(y_true, y_proba, num_classes=3):
    """Compute macro-averaged AUROC (one-vs-rest)."""
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def compute_ece(y_true, y_proba, n_bins=10):
    """Compute Expected Calibration Error."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    return float(ece / len(y_true))


def compute_per_class_metrics(y_true, y_pred):
    """Compute per-class precision, recall, and F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "f1": [float(f) for f in f1],
    }


def compute_all_metrics(y_true, y_pred, y_proba):
    """Compute all metrics in a single dict."""
    return {
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "auroc": compute_auroc(y_true, y_proba),
        "ece": compute_ece(y_true, y_proba),
        "per_class": compute_per_class_metrics(y_true, y_pred),
    }
```

**Step 3: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 8 tests PASS

**Step 4: Commit**

```bash
git add src/simtodata/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: metrics suite with F1, AUROC, ECE, per-class"
```

---

## Task 10: Smoke Test and B1 Experiment

**Files:**
- Create: `tests/test_benchmark_smoke.py`
- Create: `experiments/run_classification.py`

**Step 1: Write smoke test**

`tests/test_benchmark_smoke.py`:
```python
"""End-to-end smoke test: generate -> train -> predict -> metrics."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simtodata.data.generate import generate_dataset
from simtodata.data.transforms import Normalize
from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.train import train_model
from simtodata.models.predict import predict_batch
from simtodata.simulator.regime import RegimeConfig


def _source_regime():
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


def test_end_to_end_smoke():
    """Full pipeline smoke test with tiny data. Must complete in <60s."""
    regime = _source_regime()
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    data = generate_dataset(regime, 90, seed=42, class_distribution=class_dist)

    normalize = Normalize()
    signals = torch.from_numpy(data["signals"].astype(np.float32)).unsqueeze(1)
    labels = torch.from_numpy(data["labels"])
    for i in range(len(signals)):
        signals[i] = normalize(signals[i])

    train_ds = TensorDataset(signals[:60], labels[:60])
    val_ds = TensorDataset(signals[60:80], labels[60:80])
    test_ds = TensorDataset(signals[80:], labels[80:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = DefectCNN1D()
    model, history = train_model(model, train_loader, val_loader, epochs=2, lr=1e-3)

    preds, probs, true_labels = predict_batch(model, test_loader)

    assert preds.shape == (10,)
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert all(p in {0, 1, 2} for p in preds)
```

**Step 2: Run smoke test**

```bash
pytest tests/test_benchmark_smoke.py -v --timeout=60
```

Expected: PASS

**Step 3: Create experiment script for B1**

`experiments/run_classification.py`:
```python
"""Run classification benchmark experiments B1-B5."""

import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.models.factory import model_from_config
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model
from simtodata.evaluation.metrics import compute_all_metrics


def _save_result(name, metrics, results_dir, y_true=None, y_pred=None):
    os.makedirs(results_dir, exist_ok=True)
    result = {"name": name, "metrics": metrics}
    if y_true is not None:
        result["y_true"] = y_true.tolist()
    if y_pred is not None:
        result["y_pred"] = y_pred.tolist()
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")


def main():
    config_path = "configs/model_cnn1d.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tc = config["training"]
    ft = config["finetune"]
    results_dir = "results"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    norm = Normalize()
    source_train = InspectionDataset("data/source_train.npz", transform=norm)
    source_val = InspectionDataset("data/source_val.npz", transform=norm)
    source_test = InspectionDataset("data/source_test.npz", transform=norm)
    shifted_test = InspectionDataset("data/shifted_test.npz", transform=norm)
    adapt_data = InspectionDataset("data/shifted_adapt.npz", transform=norm)
    randomized_train = InspectionDataset("data/randomized_train.npz", transform=norm)

    bs = tc["batch_size"]
    train_loader = DataLoader(source_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(source_val, batch_size=bs)
    source_test_loader = DataLoader(source_test, batch_size=bs)
    shifted_test_loader = DataLoader(shifted_test, batch_size=bs)
    adapt_loader = DataLoader(adapt_data, batch_size=bs, shuffle=True)
    rand_train_loader = DataLoader(randomized_train, batch_size=bs, shuffle=True)

    # B1: Source -> Source
    print("B1: Training on source, eval on source...")
    model_b1 = model_from_config(config_path)
    model_b1, _ = train_model(model_b1, train_loader, val_loader, epochs=tc["epochs"],
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                              patience=tc["early_stopping_patience"],
                              scheduler_patience=tc["scheduler_patience"],
                              scheduler_factor=tc["scheduler_factor"])
    torch.save(model_b1.state_dict(), f"{models_dir}/B1_cnn1d_source.pt")
    preds, probs, labels = predict_batch(model_b1, source_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B1 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B1_cnn1d_source_on_source", metrics, results_dir, labels, preds)

    # B2: Source -> Shifted (reuse B1 model)
    print("B2: Eval B1 on shifted...")
    preds, probs, labels = predict_batch(model_b1, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B2 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B2_cnn1d_source_on_shifted", metrics, results_dir, labels, preds)

    # B3: Randomized -> Shifted
    print("B3: Training on randomized, eval on shifted...")
    model_b3 = model_from_config(config_path)
    model_b3, _ = train_model(model_b3, rand_train_loader, val_loader, epochs=tc["epochs"],
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                              patience=tc["early_stopping_patience"],
                              scheduler_patience=tc["scheduler_patience"],
                              scheduler_factor=tc["scheduler_factor"])
    torch.save(model_b3.state_dict(), f"{models_dir}/B3_cnn1d_randomized.pt")
    preds, probs, labels = predict_batch(model_b3, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B3 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B3_cnn1d_randomized_on_shifted", metrics, results_dir, labels, preds)

    # B4: Source + fine-tune -> Shifted
    print("B4: Fine-tuning B1 on adapt, eval on shifted...")
    model_b4 = model_from_config(config_path)
    model_b4.load_state_dict(torch.load(f"{models_dir}/B1_cnn1d_source.pt", weights_only=True))
    model_b4, _ = train_model(model_b4, adapt_loader, epochs=ft["epochs"],
                              lr=ft["learning_rate"])
    torch.save(model_b4.state_dict(), f"{models_dir}/B4_cnn1d_source_finetuned.pt")
    preds, probs, labels = predict_batch(model_b4, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B4 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B4_cnn1d_source_finetune_on_shifted", metrics, results_dir, labels, preds)

    # B5: Randomized + fine-tune -> Shifted
    print("B5: Fine-tuning B3 on adapt, eval on shifted...")
    model_b5 = model_from_config(config_path)
    model_b5.load_state_dict(torch.load(f"{models_dir}/B3_cnn1d_randomized.pt", weights_only=True))
    model_b5, _ = train_model(model_b5, adapt_loader, epochs=ft["epochs"],
                              lr=ft["learning_rate"])
    torch.save(model_b5.state_dict(), f"{models_dir}/B5_cnn1d_randomized_finetuned.pt")
    preds, probs, labels = predict_batch(model_b5, shifted_test_loader)
    metrics = compute_all_metrics(labels, preds, probs)
    print(f"  B5 Macro-F1: {metrics['macro_f1']:.4f}")
    _save_result("B5_cnn1d_randomized_finetune_on_shifted", metrics, results_dir, labels, preds)


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add tests/test_benchmark_smoke.py experiments/run_classification.py
git commit -m "feat: end-to-end smoke test and B1-B5 classification experiments"
```

---

## Task 11: Feature Extraction

**Files:**
- Create: `src/simtodata/features/extract.py`
- Create: `tests/test_features.py`

**Step 1: Write tests**

`tests/test_features.py`:
```python
"""Tests for hand-crafted feature extraction."""

import numpy as np
import pytest

from simtodata.features.extract import extract_features, extract_features_batch
from simtodata.simulator.forward_model import TraceParams, generate_trace


class TestExtractFeatures:
    def test_output_length(self):
        signal = np.random.randn(1024)
        features = extract_features(signal)
        assert features.shape == (11,)

    def test_finite(self):
        signal = np.random.randn(1024)
        features = extract_features(signal)
        assert np.all(np.isfinite(features))

    def test_deterministic(self):
        signal = np.random.randn(1024)
        f1 = extract_features(signal)
        f2 = extract_features(signal)
        np.testing.assert_array_equal(f1, f2)

    def test_defect_has_more_peaks(self):
        params_no = TraceParams(
            thickness_mm=20.0, velocity_ms=6000.0, attenuation_np_mm=0.02,
            center_freq_mhz=3.0, pulse_sigma_us=1.0, has_defect=False,
        )
        params_def = TraceParams(
            thickness_mm=20.0, velocity_ms=6000.0, attenuation_np_mm=0.02,
            center_freq_mhz=3.0, pulse_sigma_us=1.0, has_defect=True,
            defect_depth_mm=10.0, defect_reflectivity=0.6,
        )
        f_no = extract_features(generate_trace(params_no))
        f_def = extract_features(generate_trace(params_def))
        assert f_def[0] >= f_no[0]  # n_peaks


class TestExtractFeaturesBatch:
    def test_batch_shape(self):
        signals = np.random.randn(20, 1024)
        features = extract_features_batch(signals)
        assert features.shape == (20, 11)
```

**Step 2: Implement**

`src/simtodata/features/extract.py`:
```python
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
```

**Step 3: Run tests**

```bash
pytest tests/test_features.py -v
```

Expected: all 5 tests PASS

**Step 4: Commit**

```bash
git add src/simtodata/features/extract.py tests/test_features.py
git commit -m "feat: hand-crafted feature extraction (11 features per trace)"
```

---

## Task 12: Baseline Classifiers

**Files:**
- Create: `src/simtodata/models/baselines.py`
- Create: `tests/test_baselines.py`
- Create: `experiments/run_baselines.py`

**Step 1: Write tests**

`tests/test_baselines.py`:
```python
"""Tests for sklearn baseline classifiers."""

import numpy as np
import pytest

from simtodata.models.baselines import create_baseline


class TestBaselines:
    @pytest.fixture
    def train_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 11))
        y = rng.choice([0, 1, 2], size=200)
        return X, y

    def test_logistic_regression_fits(self, train_data):
        clf = create_baseline("logistic_regression")
        clf.fit(*train_data)
        preds = clf.predict(train_data[0][:10])
        assert preds.shape == (10,)
        assert all(p in {0, 1, 2} for p in preds)

    def test_gradient_boosting_fits(self, train_data):
        clf = create_baseline("gradient_boosting")
        clf.fit(*train_data)
        preds = clf.predict(train_data[0][:10])
        assert preds.shape == (10,)

    def test_predict_proba_valid(self, train_data):
        clf = create_baseline("gradient_boosting")
        clf.fit(*train_data)
        probs = clf.predict_proba(train_data[0][:10])
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0)

    def test_above_random(self, train_data):
        clf = create_baseline("gradient_boosting")
        clf.fit(*train_data)
        preds = clf.predict(train_data[0])
        accuracy = np.mean(preds == train_data[1])
        assert accuracy > 0.33
```

**Step 2: Implement**

`src/simtodata/models/baselines.py`:
```python
"""Non-neural baseline classifiers wrapping sklearn pipelines."""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_baseline(name: str):
    """Create a baseline classifier pipeline.

    Args:
        name: 'logistic_regression' or 'gradient_boosting'.

    Returns:
        sklearn Pipeline with StandardScaler and classifier.
    """
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
```

**Step 3: Create baselines experiment script**

`experiments/run_baselines.py`:
```python
"""Run baseline experiments B0a-B0c."""

import json
import os

import joblib
import numpy as np

from simtodata.evaluation.metrics import compute_all_metrics
from simtodata.features.extract import extract_features_batch
from simtodata.models.baselines import create_baseline


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    source_train = np.load("data/source_train.npz")
    source_test = np.load("data/source_test.npz")
    shifted_test = np.load("data/shifted_test.npz")

    print("Extracting features...")
    train_feat = extract_features_batch(source_train["signals"])
    source_test_feat = extract_features_batch(source_test["signals"])
    shifted_test_feat = extract_features_batch(shifted_test["signals"])

    # B0a: LogReg on source
    print("B0a: Logistic Regression...")
    clf_lr = create_baseline("logistic_regression")
    clf_lr.fit(train_feat, source_train["labels"])
    preds = clf_lr.predict(source_test_feat)
    probs = clf_lr.predict_proba(source_test_feat)
    metrics = compute_all_metrics(source_test["labels"], preds, probs)
    print(f"  F1: {metrics['macro_f1']:.4f}")
    with open("results/B0a_logreg_source_on_source.json", "w") as f:
        json.dump({"name": "B0a", "metrics": metrics}, f, indent=2)

    # B0b: GradBoost on source
    print("B0b: Gradient Boosting...")
    clf_gb = create_baseline("gradient_boosting")
    clf_gb.fit(train_feat, source_train["labels"])
    joblib.dump(clf_gb, "models/B0b_gb_source.joblib")
    preds = clf_gb.predict(source_test_feat)
    probs = clf_gb.predict_proba(source_test_feat)
    metrics = compute_all_metrics(source_test["labels"], preds, probs)
    print(f"  F1: {metrics['macro_f1']:.4f}")
    with open("results/B0b_gb_source_on_source.json", "w") as f:
        json.dump({"name": "B0b", "metrics": metrics}, f, indent=2)

    # B0c: GradBoost on shifted
    print("B0c: GradBoost -> shifted...")
    preds = clf_gb.predict(shifted_test_feat)
    probs = clf_gb.predict_proba(shifted_test_feat)
    metrics = compute_all_metrics(shifted_test["labels"], preds, probs)
    print(f"  F1: {metrics['macro_f1']:.4f}")
    with open("results/B0c_gb_source_on_shifted.json", "w") as f:
        json.dump({"name": "B0c", "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
pytest tests/test_baselines.py -v
```

Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add src/simtodata/models/baselines.py tests/test_baselines.py experiments/run_baselines.py
git commit -m "feat: non-neural baselines and B0a-B0c experiment script"
```

---

## Task 13: Calibration Analysis

> **Note:** Old Task 12 (Metrics Suite) was moved to Task 9 to resolve the dependency on `compute_all_metrics` in experiment scripts.

**Files:**
- Create: `src/simtodata/evaluation/calibration.py`
- Create: `tests/test_calibration.py`

**Step 1: Write tests**

`tests/test_calibration.py`:
```python
"""Tests for calibration analysis."""

import numpy as np
import pytest

from simtodata.evaluation.calibration import reliability_diagram


class TestReliabilityDiagram:
    def test_bin_counts_sum_to_n(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        _, _, counts = reliability_diagram(y_true, y_proba, n_bins=5)
        assert counts.sum() == len(y_true)

    def test_perfect_calibration(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        confidences, accuracies, _ = reliability_diagram(y_true, y_proba, n_bins=5)
        for c, a in zip(confidences, accuracies):
            if not np.isnan(c):
                assert abs(a - 1.0) < 0.01  # perfect accuracy

    def test_output_shapes(self):
        y_true = np.random.choice([0, 1, 2], 100)
        y_proba = np.random.dirichlet([1, 1, 1], 100)
        c, a, n = reliability_diagram(y_true, y_proba, n_bins=10)
        assert len(c) == 10
        assert len(a) == 10
        assert len(n) == 10
```

**Step 2: Implement**

`src/simtodata/evaluation/calibration.py`:
```python
"""Calibration analysis: reliability diagrams and ECE plotting."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reliability_diagram(y_true, y_proba, n_bins=10):
    """Compute reliability diagram data.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts) as numpy arrays of length n_bins.
        NaN for empty bins.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.full(n_bins, np.nan)
    bin_accuracies = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = mask.sum()
        if mask.sum() > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = correct[mask].mean()

    return bin_confidences, bin_accuracies, bin_counts


def plot_reliability_diagram(results_list, labels, save_path):
    """Plot reliability diagrams for multiple models.

    Args:
        results_list: List of (y_true, y_proba) tuples.
        labels: List of model names.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for (y_true, y_proba), label in zip(results_list, labels):
        conf, acc, _ = reliability_diagram(y_true, y_proba)
        valid = ~np.isnan(conf)
        ax.plot(conf[valid], acc[valid], "o-", label=label)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
```

**Step 3: Run tests**

```bash
pytest tests/test_calibration.py -v
```

Expected: all 3 tests PASS

**Step 4: Commit**

```bash
git add src/simtodata/evaluation/calibration.py tests/test_calibration.py
git commit -m "feat: calibration analysis with reliability diagrams"
```

---

## Task 14: Robustness Sweep

**Files:**
- Create: `src/simtodata/evaluation/robustness.py`
- Create: `experiments/run_robustness.py`

**Step 1: Implement**

`src/simtodata/evaluation/robustness.py`:
```python
"""Robustness sweep: evaluate models under increasing shift intensity."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simtodata.data.generate import generate_dataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.metrics import compute_all_metrics
from simtodata.models.predict import predict_batch
from simtodata.simulator.regime import RegimeConfig


INTENSITIES = {
    "none": dict(snr_db=(30, 40), baseline_drift=(0, 0), gain_variation=(1, 1),
                 jitter_samples=(0, 0), dropout_n_gaps=(0, 0), dropout_gap_length=(0, 0)),
    "low": dict(snr_db=(15, 25), baseline_drift=(0, 0.1), gain_variation=(0.9, 1.1),
                jitter_samples=(0, 1), dropout_n_gaps=(0, 1), dropout_gap_length=(5, 10)),
    "medium": dict(snr_db=(8, 15), baseline_drift=(0.1, 0.2), gain_variation=(0.8, 1.2),
                   jitter_samples=(0, 2), dropout_n_gaps=(1, 2), dropout_gap_length=(5, 15)),
    "high": dict(snr_db=(3, 8), baseline_drift=(0.2, 0.3), gain_variation=(0.7, 1.3),
                 jitter_samples=(1, 3), dropout_n_gaps=(1, 3), dropout_gap_length=(10, 20)),
    "extreme": dict(snr_db=(1, 5), baseline_drift=(0.3, 0.5), gain_variation=(0.5, 1.5),
                    jitter_samples=(2, 5), dropout_n_gaps=(2, 4), dropout_gap_length=(15, 30)),
}


def make_intensity_regime(intensity_name, base_regime=None):
    """Create a RegimeConfig for a given shift intensity."""
    params = INTENSITIES[intensity_name]
    return RegimeConfig(
        name=f"intensity_{intensity_name}",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5500.0, 6500.0),
        attenuation_np_mm=(0.01, 0.10),
        center_freq_mhz=(1.5, 7.0),
        pulse_sigma_us=(0.3, 2.0),
        defect_depth_mm=(2.0, 28.0),
        **params,
    )


def run_robustness_sweep(models, model_names, n_samples=1000, seed=42):
    """Evaluate models across shift intensity levels.

    Args:
        models: List of (model_or_pipeline, is_neural) tuples.
        model_names: List of model names for results.
        n_samples: Samples per intensity level.
        seed: Random seed.

    Returns:
        Dict mapping intensity -> model_name -> metrics.
    """
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    normalize = Normalize()
    results = {}

    for intensity_name in INTENSITIES:
        print(f"  Intensity: {intensity_name}")
        regime = make_intensity_regime(intensity_name)
        rng = np.random.default_rng(seed)
        data = generate_dataset(regime, n_samples, int(rng.integers(0, 2**31)), class_dist)
        results[intensity_name] = {}

        for (model, is_neural), name in zip(models, model_names):
            if is_neural:
                signals = torch.from_numpy(data["signals"].astype(np.float32)).unsqueeze(1)
                labels = torch.from_numpy(data["labels"])
                for i in range(len(signals)):
                    signals[i] = normalize(signals[i])
                loader = DataLoader(TensorDataset(signals, labels), batch_size=256)
                preds, probs, true_labels = predict_batch(model, loader)
            else:
                from simtodata.features.extract import extract_features_batch
                features = extract_features_batch(data["signals"])
                preds = model.predict(features)
                probs = model.predict_proba(features)
                true_labels = data["labels"]

            metrics = compute_all_metrics(true_labels, preds, probs)
            results[intensity_name][name] = metrics
            print(f"    {name}: F1={metrics['macro_f1']:.4f}")

    return results
```

`experiments/run_robustness.py`:
```python
"""Run robustness sweep across shift intensities."""

import json
import os

import joblib
import torch

from simtodata.evaluation.robustness import run_robustness_sweep
from simtodata.models.factory import model_from_config


def main():
    os.makedirs("results", exist_ok=True)

    config_path = "configs/model_cnn1d.yaml"

    # Load models
    models = []
    names = []

    # B0c: Gradient Boosting
    if os.path.exists("models/B0b_gb_source.joblib"):
        clf = joblib.load("models/B0b_gb_source.joblib")
        models.append((clf, False))
        names.append("B0c_gb")

    # B2: Source CNN
    if os.path.exists("models/B1_cnn1d_source.pt"):
        model_b2 = model_from_config(config_path)
        model_b2.load_state_dict(torch.load("models/B1_cnn1d_source.pt", weights_only=True))
        models.append((model_b2, True))
        names.append("B2_cnn1d_source")

    # B5: Randomized + finetuned CNN
    if os.path.exists("models/B5_cnn1d_randomized_finetuned.pt"):
        model_b5 = model_from_config(config_path)
        model_b5.load_state_dict(
            torch.load("models/B5_cnn1d_randomized_finetuned.pt", weights_only=True)
        )
        models.append((model_b5, True))
        names.append("B5_cnn1d_rand_ft")

    print("Running robustness sweep...")
    results = run_robustness_sweep(models, names)

    with open("results/robustness_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/robustness_sweep.json")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/simtodata/evaluation/robustness.py experiments/run_robustness.py
git commit -m "feat: robustness sweep across 5 shift intensity levels"
```

---

## Task 15: Adaptation Efficiency Curve

**Files:**
- Create: `src/simtodata/evaluation/adaptation_curve.py`
- Create: `experiments/run_adaptation_curve.py`

**Step 1: Implement**

`src/simtodata/evaluation/adaptation_curve.py`:
```python
"""Adaptation efficiency: fine-tune sample count vs performance."""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.metrics import compute_macro_f1
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model


def run_adaptation_sweep(model_template, pretrained_path, adapt_dataset, eval_loader,
                         sample_counts=(0, 25, 50, 100, 200, 500), n_repeats=3,
                         ft_epochs=20, ft_lr=1e-4):
    """Run adaptation efficiency sweep.

    Args:
        model_template: Uninitialized model (for architecture).
        pretrained_path: Path to pretrained weights.
        adapt_dataset: Full adaptation InspectionDataset.
        eval_loader: DataLoader for evaluation.
        sample_counts: List of fine-tune sample counts to try.
        n_repeats: Repeats per count for error bars.
        ft_epochs: Fine-tuning epochs.
        ft_lr: Fine-tuning learning rate.

    Returns:
        Dict mapping count -> {'mean_f1': float, 'std_f1': float}.
    """
    results = {}
    for count in sample_counts:
        f1_scores = []
        for repeat in range(n_repeats):
            model = copy.deepcopy(model_template)
            model.load_state_dict(torch.load(pretrained_path, weights_only=True))

            if count == 0:
                # No fine-tuning, just evaluate
                preds, _, labels = predict_batch(model, eval_loader)
                f1_scores.append(compute_macro_f1(labels, preds))
                break  # No variance for 0 samples
            else:
                rng = np.random.default_rng(42 + repeat)
                indices = rng.choice(len(adapt_dataset), size=min(count, len(adapt_dataset)),
                                     replace=False)
                subset = Subset(adapt_dataset, indices.tolist())
                ft_loader = DataLoader(subset, batch_size=min(64, count), shuffle=True)
                model, _ = train_model(model, ft_loader, epochs=ft_epochs, lr=ft_lr)
                preds, _, labels = predict_batch(model, eval_loader)
                f1_scores.append(compute_macro_f1(labels, preds))

        results[count] = {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)) if len(f1_scores) > 1 else 0.0,
        }
        print(f"  {count} samples: F1 = {results[count]['mean_f1']:.4f} +/- {results[count]['std_f1']:.4f}")

    return results
```

`experiments/run_adaptation_curve.py`:
```python
"""Run adaptation efficiency experiment."""

import json
import os

import torch
from torch.utils.data import DataLoader

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.adaptation_curve import run_adaptation_sweep
from simtodata.models.factory import model_from_config


def main():
    os.makedirs("results", exist_ok=True)
    config_path = "configs/model_cnn1d.yaml"
    norm = Normalize()

    adapt_data = InspectionDataset("data/shifted_adapt.npz", transform=norm)
    shifted_test = InspectionDataset("data/shifted_test.npz", transform=norm)
    eval_loader = DataLoader(shifted_test, batch_size=256)

    model_template = model_from_config(config_path)
    all_results = {}

    # Source-pretrained (B4-style)
    if os.path.exists("models/B1_cnn1d_source.pt"):
        print("Adaptation curve: source-pretrained...")
        results = run_adaptation_sweep(
            model_template, "models/B1_cnn1d_source.pt", adapt_data, eval_loader
        )
        all_results["source_pretrained"] = results

    # Randomized-pretrained (B5-style)
    if os.path.exists("models/B3_cnn1d_randomized.pt"):
        print("Adaptation curve: randomized-pretrained...")
        results = run_adaptation_sweep(
            model_template, "models/B3_cnn1d_randomized.pt", adapt_data, eval_loader
        )
        all_results["randomized_pretrained"] = results

    with open("results/adaptation_curve.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved: results/adaptation_curve.json")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/simtodata/evaluation/adaptation_curve.py experiments/run_adaptation_curve.py
git commit -m "feat: adaptation efficiency curve with sample count sweep"
```

---

## Task 16: Visualization and Figures

**Files:**
- Create: `experiments/generate_figures.py`

**Step 1: Implement**

`experiments/generate_figures.py`:
```python
"""Generate all figures from result JSONs."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simtodata.data.generate import generate_dataset
from simtodata.simulator.regime import RegimeConfig


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_robustness_curve(results_path, save_path):
    """Plot F1 vs shift intensity for multiple models."""
    results = _load_json(results_path)
    intensities = list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(results[intensities[0]].keys())

    for name in model_names:
        f1s = [results[intensity][name]["macro_f1"] for intensity in intensities]
        ax.plot(range(len(intensities)), f1s, "o-", label=name)

    ax.set_xticks(range(len(intensities)))
    ax.set_xticklabels(intensities)
    ax.set_xlabel("Shift Intensity")
    ax.set_ylabel("Macro F1")
    ax.set_title("Robustness Under Increasing Domain Shift")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_adaptation_curve(results_path, save_path):
    """Plot F1 vs fine-tune sample count."""
    results = _load_json(results_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, data in results.items():
        counts = sorted([int(k) for k in data.keys()])
        means = [data[str(c)]["mean_f1"] for c in counts]
        stds = [data[str(c)]["std_f1"] for c in counts]
        ax.errorbar(counts, means, yerr=stds, fmt="o-", label=strategy, capsize=3)

    ax.set_xlabel("Fine-tune Samples")
    ax.set_ylabel("Macro F1 on Shifted Test")
    ax.set_title("Adaptation Efficiency")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_example_traces(save_path):
    """Plot example traces: 3 classes x 2 regimes."""
    source = RegimeConfig("source", (15, 25), (5800, 6200), (0.01, 0.05), (2, 5),
                          (0.5, 1.5), (2, 28), (25, 35), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0))
    shifted = RegimeConfig("shifted", (15, 25), (5500, 6500), (0.01, 0.10), (1.5, 7),
                           (0.3, 2), (2, 28), (8, 15), (0.1, 0.2), (0.8, 1.2), (0, 2),
                           (1, 2), (5, 15))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    class_names = ["No Defect", "Low Severity", "High Severity"]
    regime_names = ["Source", "Shifted"]
    class_dist = {"no_defect": 1.0, "low_severity": 0.0, "high_severity": 0.0}

    for row, (regime, regime_name) in enumerate([(source, "Source"), (shifted, "Shifted")]):
        for col, cls_name in enumerate(class_names):
            dists = [
                {"no_defect": 1.0, "low_severity": 0.0, "high_severity": 0.0},
                {"no_defect": 0.0, "low_severity": 1.0, "high_severity": 0.0},
                {"no_defect": 0.0, "low_severity": 0.0, "high_severity": 1.0},
            ]
            data = generate_dataset(regime, 1, seed=42 + col, class_distribution=dists[col])
            axes[row, col].plot(data["signals"][0], linewidth=0.5)
            axes[row, col].set_title(f"{regime_name} - {cls_name}")
            axes[row, col].set_xlabel("Sample")
            axes[row, col].set_ylabel("Amplitude")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results_dir, save_path):
    """Plot confusion matrices for B1, B2, B5 (requires saved y_true/y_pred)."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    exp_names = ["B1_cnn1d_source_on_source", "B2_cnn1d_source_on_shifted",
                 "B5_cnn1d_randomized_finetune_on_shifted"]
    titles = ["B1: Source→Source", "B2: Source→Shifted", "B5: Rand+FT→Shifted"]
    class_labels = ["No Defect", "Low", "High"]

    for ax, exp_name, title in zip(axes, exp_names, titles):
        path = os.path.join(results_dir, f"{exp_name}.json")
        if not os.path.exists(path):
            ax.set_title(f"{title}\n(not available)")
            continue
        result = _load_json(path)
        if "y_true" in result and "y_pred" in result:
            cm = confusion_matrix(result["y_true"], result["y_pred"], normalize="true")
            disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
            disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
            ax.set_title(title)
        else:
            # Fallback: plot per-class recall bars
            per_class = result["metrics"]["per_class"]
            recalls = per_class["recall"]
            ax.bar(class_labels, recalls)
            ax.set_ylabel("Recall")
            ax.set_title(title)
            ax.set_ylim(0, 1)
            for i, v in enumerate(recalls):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs("docs/figures", exist_ok=True)

    if os.path.exists("results/robustness_sweep.json"):
        print("Plotting robustness curve...")
        plot_robustness_curve("results/robustness_sweep.json", "docs/figures/robustness_curve.png")

    if os.path.exists("results/adaptation_curve.json"):
        print("Plotting adaptation curve...")
        plot_adaptation_curve("results/adaptation_curve.json", "docs/figures/adaptation_curve.png")

    print("Plotting example traces...")
    plot_example_traces("docs/figures/example_traces.png")

    print("Plotting confusion matrices...")
    plot_confusion_matrices("results", "docs/figures/confusion_matrices.png")

    print("All figures generated.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add experiments/generate_figures.py
git commit -m "feat: figure generation for robustness, adaptation, traces, confusion"
```

---

## Task 17: Run-All Script, CI, and Makefile

**Files:**
- Create: `experiments/run_all.py`
- Create: `.github/workflows/ci.yml`
- Create: `Makefile`

**Step 1: Implement `run_all.py`**

`experiments/run_all.py`:
```python
"""Single entry point for the full benchmark pipeline."""

import argparse
import subprocess
import sys
import time


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable] + cmd.split(), cwd=".")
    if result.returncode != 0:
        print(f"FAILED: {description}")
        sys.exit(1)
    print(f"\n  Completed in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small datasets, 2 epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen_args = "-m simtodata.data.generate"
    if args.quick:
        gen_args += " --quick"

    run(gen_args, "Generating datasets")
    run("experiments/run_baselines.py", "Running baselines B0a-B0c")
    run("experiments/run_classification.py", "Running CNN experiments B1-B5")
    run("experiments/run_robustness.py", "Running robustness sweep")
    run("experiments/run_adaptation_curve.py", "Running adaptation curve")
    run("experiments/generate_figures.py", "Generating figures")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Create CI**

`.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: pytest tests/ -v --tb=short
```

**Step 3: Create Makefile**

```makefile
.PHONY: generate train-baselines train-cnn evaluate figures all test lint clean

generate:
	python -m simtodata.data.generate

train-baselines:
	python experiments/run_baselines.py

train-cnn:
	python experiments/run_classification.py

evaluate:
	python experiments/run_robustness.py
	python experiments/run_adaptation_curve.py

figures:
	python experiments/generate_figures.py

all: generate train-baselines train-cnn evaluate figures

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf results/ data/ models/ docs/figures/*.png
```

**Step 4: Commit**

```bash
git add experiments/run_all.py .github/workflows/ci.yml Makefile
git commit -m "feat: run-all script, CI pipeline, and Makefile"
```

---

## Task 18: Test Hardening

**Goal:** Push total test count to 50+ by adding edge cases and shared fixtures.

**Files:**
- Create: `tests/conftest.py` — shared fixtures to eliminate duplicated `_source_regime()`
- Modify: `tests/test_forward_model.py` — add edge case tests
- Modify: `tests/test_noise.py` — add edge case tests
- Create: `tests/test_pipeline_integration.py`

**Step 1: Create shared test fixtures**

`tests/conftest.py`:
```python
"""Shared test fixtures."""

import pytest
from simtodata.simulator.regime import RegimeConfig


@pytest.fixture
def source_regime():
    """Source regime config for testing."""
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )
```

After creating conftest.py, refactor test files that define their own `_source_regime()` to use the shared `source_regime` fixture instead.

**Step 2: Add edge case tests to `test_forward_model.py`**

Append:
```python
class TestEdgeCases:
    def test_defect_at_surface(self):
        """Defect near surface should not crash."""
        params = _default_params(has_defect=True, defect_depth_mm=0.5, defect_reflectivity=0.5)
        signal = generate_trace(params)
        assert signal.shape == (1024,)
        assert np.all(np.isfinite(signal))

    def test_defect_near_backwall(self):
        """Defect near back wall should not crash."""
        params = _default_params(has_defect=True, defect_depth_mm=19.0, defect_reflectivity=0.5)
        signal = generate_trace(params)
        assert signal.shape == (1024,)

    def test_zero_attenuation(self):
        """Zero attenuation means no amplitude decay."""
        params = _default_params(attenuation_np_mm=0.0)
        signal = generate_trace(params)
        # Surface and back-wall echoes should have similar peak amplitude
        abs_signal = np.abs(signal)
        peaks, props = find_peaks(abs_signal, height=0.1 * np.max(abs_signal))
        if len(peaks) >= 2:
            sorted_heights = sorted(props["peak_heights"], reverse=True)
            assert sorted_heights[1] / sorted_heights[0] > 0.8

    def test_high_attenuation(self):
        """High attenuation makes back-wall echo nearly invisible."""
        params = _default_params(attenuation_np_mm=0.2)
        signal = generate_trace(params)
        assert np.all(np.isfinite(signal))
```

**Step 2: Add edge case tests to `test_noise.py`**

Append:
```python
class TestEdgeCases:
    def test_snr_zero_db(self):
        signal = np.sin(np.linspace(0, 10, 1024))
        rng = np.random.default_rng(42)
        noisy = add_gaussian_noise(signal, 0.0, rng)
        assert np.all(np.isfinite(noisy))

    def test_dropout_covers_half_signal(self):
        signal = np.ones(1024)
        rng = np.random.default_rng(42)
        result = add_masked_dropout(signal, 10, (50, 60), rng)
        assert np.all(np.isfinite(result))
        assert np.sum(result == 0) > 0

    def test_max_jitter(self):
        signal = np.ones(1024)
        rng = np.random.default_rng(42)
        result = add_temporal_jitter(signal, 10, rng)
        assert np.all(np.isfinite(result))
        assert result.shape == (1024,)
```

**Step 3: Create integration test**

`tests/test_pipeline_integration.py`:
```python
"""Integration tests: config -> generate -> train -> evaluate."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simtodata.data.generate import generate_dataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.metrics import compute_all_metrics
from simtodata.features.extract import extract_features_batch
from simtodata.models.baselines import create_baseline
from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model
from simtodata.simulator.regime import RegimeConfig


def _source_regime():
    return RegimeConfig(
        name="source", thickness_mm=(10, 30), velocity_ms=(5800, 6200),
        attenuation_np_mm=(0.01, 0.05), center_freq_mhz=(2, 5),
        pulse_sigma_us=(0.5, 1.5), defect_depth_mm=(2, 28),
        snr_db=(20, 40), baseline_drift=(0, 0), gain_variation=(1, 1),
        jitter_samples=(0, 0), dropout_n_gaps=(0, 0), dropout_gap_length=(0, 0),
    )


def test_baseline_pipeline():
    """Generate -> extract features -> fit baseline -> predict -> metrics."""
    regime = _source_regime()
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    data = generate_dataset(regime, 300, seed=42, class_distribution=class_dist)

    features = extract_features_batch(data["signals"])
    assert features.shape == (300, 11)

    clf = create_baseline("gradient_boosting")
    clf.fit(features[:200], data["labels"][:200])
    preds = clf.predict(features[200:])
    probs = clf.predict_proba(features[200:])
    metrics = compute_all_metrics(data["labels"][200:], preds, probs)

    assert "macro_f1" in metrics
    assert 0 <= metrics["macro_f1"] <= 1
    assert np.isfinite(metrics["ece"])


def test_cnn_pipeline():
    """Generate -> dataset -> train CNN -> predict -> metrics."""
    regime = _source_regime()
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    data = generate_dataset(regime, 120, seed=42, class_distribution=class_dist)

    normalize = Normalize()
    signals = torch.from_numpy(data["signals"].astype(np.float32)).unsqueeze(1)
    labels = torch.from_numpy(data["labels"])
    for i in range(len(signals)):
        signals[i] = normalize(signals[i])

    train_ds = TensorDataset(signals[:80], labels[:80])
    test_ds = TensorDataset(signals[80:], labels[80:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = DefectCNN1D()
    model, history = train_model(model, train_loader, epochs=3, lr=1e-3)

    preds, probs, true_labels = predict_batch(model, test_loader)
    metrics = compute_all_metrics(true_labels, preds, probs)

    assert "macro_f1" in metrics
    assert "auroc" in metrics
    assert len(history["train_loss"]) == 3
```

**Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: 50+ tests PASS

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: edge cases and integration tests, 50+ total"
```

---

## Task 19: Spectrogram Model (CUTTABLE — Day 6)

> Skip this task if behind schedule. The project is shippable without it.

**Files:**
- Modify: `src/simtodata/data/transforms.py` — add SpectrogramTransform
- Create: `src/simtodata/models/cnn2d_spectrogram.py`
- Modify: `src/simtodata/models/factory.py` — register cnn_2d_spectrogram
- Create: `configs/model_spec2d.yaml`
- Create: `tests/test_model_cnn2d.py`
- Create: `tests/test_transforms.py`

**Step 1: Add SpectrogramTransform**

Add to `src/simtodata/data/transforms.py`:
```python
import numpy as np
from scipy.signal import stft as scipy_stft


class SpectrogramTransform:
    """STFT spectrogram transform for 2D CNN input."""

    def __init__(self, window_size=64, hop_length=16, fs=50e6):
        self.window_size = window_size
        self.hop_length = hop_length
        self.fs = fs

    def __call__(self, signal):
        """Transform (1, 1024) tensor to (1, n_freq, n_time) spectrogram."""
        import torch
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
```

**Step 2: Create 2D CNN**

`src/simtodata/models/cnn2d_spectrogram.py`:
```python
"""2D CNN classifier on STFT spectrograms."""

import torch.nn as nn


class DefectCNN2D(nn.Module):
    """2D convolutional network for spectrogram classification."""

    def __init__(self, channels=(16, 32, 64), kernels=(3, 3, 3), fc_hidden=32,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], kernels[0]),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernels[1]),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernels[2]),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels[2], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
```

**Step 3: Update factory**

Add to `src/simtodata/models/factory.py`:
```python
from simtodata.models.cnn2d_spectrogram import DefectCNN2D

# Add inside model_from_config, after cnn_1d block:
    elif arch["type"] == "cnn_2d_spectrogram":
        return DefectCNN2D(
            channels=tuple(arch["channels"]),
            kernels=tuple(arch["kernels"]),
            fc_hidden=arch["fc_hidden"],
            dropout=arch["dropout"],
            num_classes=arch["num_classes"],
        )
```

**Step 4: Config**

`configs/model_spec2d.yaml`:
```yaml
architecture:
  type: cnn_2d_spectrogram
  channels: [16, 32, 64]
  kernels: [3, 3, 3]
  fc_hidden: 32
  dropout: 0.3
  num_classes: 3

training:
  epochs: 50
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler_patience: 5
  scheduler_factor: 0.5
  early_stopping_patience: 10

finetune:
  epochs: 20
  learning_rate: 0.0001
```

**Step 5: Tests**

`tests/test_transforms.py`:
```python
"""Tests for signal transforms."""

import numpy as np
import torch
import pytest

from simtodata.data.transforms import Normalize, SpectrogramTransform


class TestSpectrogramTransform:
    def test_output_shape(self):
        signal = torch.randn(1, 1024)
        transform = SpectrogramTransform(window_size=64, hop_length=16)
        spec = transform(signal)
        assert spec.dim() == 3  # (1, n_freq, n_time)
        assert spec.shape[0] == 1

    def test_finite(self):
        signal = torch.randn(1, 1024)
        transform = SpectrogramTransform()
        spec = transform(signal)
        assert torch.all(torch.isfinite(spec))

    def test_deterministic(self):
        signal = torch.randn(1, 1024)
        transform = SpectrogramTransform()
        s1 = transform(signal)
        s2 = transform(signal)
        torch.testing.assert_close(s1, s2)
```

`tests/test_model_cnn2d.py`:
```python
"""Tests for 2D CNN spectrogram classifier."""

import torch
from simtodata.models.cnn2d_spectrogram import DefectCNN2D


class TestDefectCNN2D:
    def test_output_shape(self):
        model = DefectCNN2D()
        x = torch.randn(8, 1, 33, 61)
        out = model(x)
        assert out.shape == (8, 3)

    def test_gradients_flow(self):
        model = DefectCNN2D()
        x = torch.randn(4, 1, 33, 61)
        out = model(x)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient: {name}"

    def test_from_config(self):
        from simtodata.models.factory import model_from_config
        model = model_from_config("configs/model_spec2d.yaml")
        assert isinstance(model, DefectCNN2D)
```

**Step 6: Run tests and commit**

```bash
pytest tests/test_transforms.py tests/test_model_cnn2d.py -v
git add src/simtodata/data/transforms.py src/simtodata/models/cnn2d_spectrogram.py \
        src/simtodata/models/factory.py configs/model_spec2d.yaml \
        tests/test_transforms.py tests/test_model_cnn2d.py
git commit -m "feat: spectrogram CNN and STFT transform (optional Day 6)"
```

---

## Task 20: README and Ship

**Files:**
- Create: `README.md`

**Step 1: Write README**

The README should follow the template from the design doc (Section 9 of the original plan). Key sections:

1. One-sentence summary
2. Problem statement
3. Approach (4 bullets: Simulate, Train, Shift, Adapt)
4. Results (Table 1 populated with actual numbers from `results/`)
5. Key Findings (fill from actual experiment results)
6. Robustness figure
7. Adaptation figure
8. Honest Scope section
9. Quick Start
10. Engineering (test count, CI, Makefile)
11. Project Structure

**Important:** Fill all table cells with real numbers from `results/*.json`. Do not leave placeholders.

**Step 2: Verify everything**

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest tests/ -v
python experiments/run_all.py --quick
```

All must pass.

**Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: README with results, figures, and honest scope"
git push origin main
```

---

## Test Count Summary

| Test File | Count |
|-----------|-------|
| test_forward_model.py | 15 |
| test_noise.py | 19 |
| test_regime.py | 6 |
| test_dataset.py | 9 |
| test_model_cnn1d.py | 6 |
| test_features.py | 5 |
| test_baselines.py | 4 |
| test_metrics.py | 8 |
| test_calibration.py | 3 |
| test_benchmark_smoke.py | 1 |
| test_pipeline_integration.py | 2 |
| test_transforms.py* | 3 |
| test_model_cnn2d.py* | 3 |
| **Total** | **84** |

*Only if Task 19 (spectrogram) is executed.

---

## Day-to-Task Mapping

```
Day 1:  Tasks 1-3    (scaffold, forward model, noise)
Day 2:  Tasks 4-6    (regime, data generation, dataset)
Day 3:  Tasks 7-10   (CNN, training, metrics, smoke test + B1)
Day 4:  Tasks 11-12  (features, baselines) + run B0-B5 experiments
Day 5:  Tasks 13-14  (calibration, robustness sweep)
Day 6:  Task 19      (spectrogram — CUTTABLE)
Day 7:  Tasks 15-16  (adaptation curve, figures)
Day 8:  Tasks 17-18  (CI, Makefile, test hardening)
Day 9:  Task 20      (README, ship)
```

## Expected Pattern to Verify After B0-B5

```
B1 (source→source)      > B2 (source→shifted)        ← shift hurts
B3 (randomized→shifted)  > B2 (source→shifted)        ← randomization helps
B4 (source+ft→shifted)   > B2 (source→shifted)        ← fine-tuning helps
B5 (rand+ft→shifted)    ≥ B3 and B4                   ← best combined
B0b (GB→source)          comparable to B1              ← CNN justified
B0c (GB→shifted)         comparable to B2              ← shift affects all
```

If any inequality breaks, refer to the Risk Register in the design doc.
