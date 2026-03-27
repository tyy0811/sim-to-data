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
