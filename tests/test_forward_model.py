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
        # LOW_REFLECTIVITY_RANGE upper is 0.3, HIGH lower is 0.4
        # Values in gap (0.3, 0.4) classify as low (below HIGH threshold)
        assert classify_severity(0.3) == 1
        assert classify_severity(0.35) == 1  # gap value -> low
        assert classify_severity(0.4) == 2   # at HIGH threshold -> high
        assert classify_severity(0.05) == 1  # below LOW range but > 0 -> low


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

    def test_depth_clamped_by_thickness(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            d = sample_defect(rng, 2, (2.0, 28.0), thickness_mm=10.0)
            assert d.depth_mm < 10.0

    def test_generate_trace_rejects_invalid_depth(self):
        params = _default_params(
            has_defect=True, defect_depth_mm=25.0, defect_reflectivity=0.5
        )
        with pytest.raises(ValueError, match="defect_depth_mm"):
            generate_trace(params)


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


class TestDefectConfigPositionMm:
    def test_default_is_none(self):
        from simtodata.simulator.defects import DefectConfig
        d = DefectConfig(depth_mm=10.0, reflectivity=0.5, severity_label=2)
        assert d.position_mm is None

    def test_explicit_position(self):
        from simtodata.simulator.defects import DefectConfig
        d = DefectConfig(depth_mm=10.0, reflectivity=0.5, severity_label=2, position_mm=42.0)
        assert d.position_mm == 42.0

    def test_sample_defect_still_works(self):
        """sample_defect() must not break — it doesn't pass position_mm."""
        import numpy as np
        from simtodata.simulator.defects import sample_defect
        rng = np.random.default_rng(42)
        d = sample_defect(rng, severity=1, depth_range=(2.0, 28.0))
        assert d.position_mm is None
        assert d.depth_mm > 0
