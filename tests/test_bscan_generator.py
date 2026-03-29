"""Tests for synthetic B-scan generation."""

import numpy as np
import pytest

from simtodata.simulator.bscan import (
    BscanResult,
    generate_synthetic_bscan,
    generate_bscan_dataset,
)
from simtodata.simulator.regime import load_regimes_from_yaml


@pytest.fixture
def source_regime():
    """Source regime for B-scan tests (reuse conftest pattern)."""
    from simtodata.simulator.regime import RegimeConfig
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        defect_reflectivity=(0.1, 0.8),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


class TestBscanResult:
    def test_is_namedtuple(self):
        r = BscanResult(bscan=np.zeros((4, 8)), label=0, mask=None)
        assert r.bscan.shape == (4, 8)
        assert r.label == 0
        assert r.mask is None


class TestGenerateSyntheticBscan:
    def test_output_shape(self, source_regime):
        rng = np.random.default_rng(42)
        result = generate_synthetic_bscan(source_regime, rng, n_positions=32)
        assert result.bscan.shape == (32, 1024)

    def test_label_is_binary(self, source_regime):
        rng = np.random.default_rng(42)
        result = generate_synthetic_bscan(source_regime, rng)
        assert result.label in (0, 1)

    def test_mask_none_by_default(self, source_regime):
        rng = np.random.default_rng(42)
        result = generate_synthetic_bscan(source_regime, rng)
        assert result.mask is None

    def test_mask_computed_when_requested(self, source_regime):
        rng = np.random.default_rng(42)
        # Force a defect to be present
        from simtodata.simulator.defects import DefectConfig
        defect = DefectConfig(
            depth_mm=15.0, reflectivity=0.6, severity_label=2, position_mm=32.0,
        )
        result = generate_synthetic_bscan(
            source_regime, rng, n_positions=64, defects=[defect], return_mask=True,
        )
        assert result.mask is not None
        assert result.mask.shape == result.bscan.shape
        assert result.mask.dtype == bool
        assert result.mask.any()  # at least some pixels marked

    def test_no_defect_bscan_has_zero_mask(self, source_regime):
        rng = np.random.default_rng(42)
        result = generate_synthetic_bscan(
            source_regime, rng, n_positions=32, defects=[], return_mask=True,
        )
        assert result.label == 0
        assert result.mask is not None
        assert not result.mask.any()  # no defect -> all False

    def test_deterministic(self, source_regime):
        r1 = generate_synthetic_bscan(source_regime, np.random.default_rng(42))
        r2 = generate_synthetic_bscan(source_regime, np.random.default_rng(42))
        np.testing.assert_array_equal(r1.bscan, r2.bscan)
        assert r1.label == r2.label

    def test_flaw_bscan_has_higher_mid_energy(self, source_regime):
        """Flaw B-scan should have extra energy between surface and back-wall."""
        rng = np.random.default_rng(42)
        from simtodata.simulator.defects import DefectConfig
        defect = DefectConfig(
            depth_mm=15.0, reflectivity=0.7, severity_label=2, position_mm=32.0,
        )
        flaw = generate_synthetic_bscan(
            source_regime, rng, n_positions=64, defects=[defect],
        )
        noflaw = generate_synthetic_bscan(
            source_regime, np.random.default_rng(42), n_positions=64, defects=[],
        )
        # Mid-region energy (between surface and back-wall) should be higher for flaw
        mid = slice(64, 900)  # rough mid-region in samples
        flaw_energy = np.sum(flaw.bscan[:, mid] ** 2)
        noflaw_energy = np.sum(noflaw.bscan[:, mid] ** 2)
        assert flaw_energy > noflaw_energy

    def test_finite(self, source_regime):
        rng = np.random.default_rng(42)
        result = generate_synthetic_bscan(source_regime, rng)
        assert np.all(np.isfinite(result.bscan))

    def test_mask_aligned_with_jitter(self):
        """Mask should cover defect echo region even when jitter shifts the trace."""
        from simtodata.simulator.regime import RegimeConfig
        from simtodata.simulator.defects import DefectConfig

        regime = RegimeConfig(
            name="jittery",
            thickness_mm=(20.0, 20.0),
            velocity_ms=(6000.0, 6000.0),
            attenuation_np_mm=(0.02, 0.02),
            center_freq_mhz=(3.0, 3.0),
            pulse_sigma_us=(1.0, 1.0),
            defect_depth_mm=(5.0, 15.0),
            defect_reflectivity=(0.6, 0.6),
            snr_db=(40.0, 40.0),
            baseline_drift=(0.0, 0.0),
            gain_variation=(1.0, 1.0),
            jitter_samples=(10, 10),  # guaranteed non-zero jitter
            dropout_n_gaps=(0, 0),
            dropout_gap_length=(0, 0),
        )
        defect = DefectConfig(
            depth_mm=10.0, reflectivity=0.6, severity_label=2, position_mm=16.0,
        )
        rng = np.random.default_rng(42)
        result = generate_synthetic_bscan(
            regime, rng, n_positions=32, defects=[defect], return_mask=True,
        )
        # The mask should cover the defect echo region (not the surface echo).
        # Defect at 10mm, v=6000m/s: arrival ~4.3us -> sample ~217, ±10 jitter.
        # Surface echo is near sample ~50, so mask center far from surface
        # confirms the mask tracks the shifted defect, not the clean position.
        pos = 16
        assert result.mask[pos].any(), "Mask should have True values at beam center"
        masked = np.where(result.mask[pos])[0]
        mask_center = float(np.mean(masked))
        assert 180 < mask_center < 260, (
            f"Mask center at {mask_center:.0f}, expected near defect echo (~217)"
        )

    def test_defect_without_position_labeled_zero(self, source_regime):
        """A defect with position_mm=None should not count as flaw."""
        from simtodata.simulator.defects import DefectConfig
        rng = np.random.default_rng(42)
        defect = DefectConfig(depth_mm=10.0, reflectivity=0.5, severity_label=2)
        result = generate_synthetic_bscan(
            source_regime, rng, n_positions=32, defects=[defect],
        )
        assert result.label == 0, "Defect without position_mm should be labeled 0"

    def test_off_scan_defect_labeled_zero(self, source_regime):
        """A defect positioned outside the scan range should be labeled 0."""
        from simtodata.simulator.defects import DefectConfig
        rng = np.random.default_rng(42)
        defect = DefectConfig(
            depth_mm=10.0, reflectivity=0.5, severity_label=2, position_mm=1000.0,
        )
        result = generate_synthetic_bscan(
            source_regime, rng, n_positions=32, defects=[defect],
        )
        assert result.label == 0, "Off-scan defect should be labeled 0"
        # B-scan should be identical to a no-defect scan with the same rng
        rng2 = np.random.default_rng(42)
        noflaw = generate_synthetic_bscan(
            source_regime, rng2, n_positions=32, defects=[],
        )
        np.testing.assert_array_equal(result.bscan, noflaw.bscan)


class TestGenerateBscanDataset:
    def test_shapes(self, source_regime):
        data = generate_bscan_dataset(
            source_regime, n_samples=10, seed=42, n_positions=32,
        )
        assert data["bscans"].shape == (10, 32, 1024)
        assert data["labels"].shape == (10,)

    def test_class_balance(self, source_regime):
        data = generate_bscan_dataset(
            source_regime, n_samples=200, seed=42, flaw_ratio=0.5,
        )
        n_flaw = (data["labels"] == 1).sum()
        observed = n_flaw / len(data["labels"])
        assert 0.35 < observed < 0.65, f"flaw_ratio=0.5 but got {observed:.3f}"

    def test_flaw_ratio_honored(self, source_regime):
        """flaw_ratio should control actual positive rate without compounding."""
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            data = generate_bscan_dataset(
                source_regime, n_samples=200, seed=42, flaw_ratio=ratio,
                n_positions=16,
            )
            observed = (data["labels"] == 1).mean()
            assert abs(observed - ratio) < 0.15, (
                f"flaw_ratio={ratio} but observed={observed:.3f}"
            )

    def test_labels_binary(self, source_regime):
        data = generate_bscan_dataset(source_regime, n_samples=20, seed=42)
        assert set(np.unique(data["labels"])).issubset({0, 1})
