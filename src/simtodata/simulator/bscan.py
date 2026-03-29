"""Synthetic B-scan generation: stack A-scans with spatial defect model.

Generates 2D B-scan images by scanning a transducer across the material surface.
Each scan position produces one A-scan; defect echo amplitude varies with distance
from the beam center (Gaussian profile).

Designed for reuse by Extension 6 (segmentation): the `defects` parameter accepts
an explicit list with position_mm, and `return_mask=True` produces a pixel-level
defect mask.
"""

from __future__ import annotations

from copy import copy
from dataclasses import asdict
from typing import NamedTuple

import numpy as np

from simtodata.simulator.defects import DefectConfig, sample_defect
from simtodata.simulator.forward_model import TraceParams, generate_trace
from simtodata.simulator.noise import apply_all_noise
from simtodata.simulator.regime import RegimeConfig, sample_trace_params


class BscanResult(NamedTuple):
    """Result of B-scan generation.

    bscan: (n_positions, n_samples) array of stacked A-scans.
    label: 0 (no flaw) or 1 (flaw present).
    mask: (n_positions, n_samples) boolean array marking defect echo pixels,
          or None when return_mask=False. When return_mask=False, result.mask
          is None. When True, mask is computed from echo thresholding.
    """

    bscan: np.ndarray
    label: int
    mask: np.ndarray | None


def _copy_trace_params(params: TraceParams) -> TraceParams:
    """Create an independent copy of TraceParams."""
    return TraceParams(**asdict(params))


def _beam_sensitivity(position: float, defect_position: float,
                      beam_width: float) -> float:
    """Gaussian beam profile: sensitivity falls off with distance from center."""
    distance = abs(position - defect_position)
    sigma = beam_width / 2.5  # beam_width ~ 2.5 sigma
    return float(np.exp(-distance**2 / (2 * sigma**2)))


def generate_synthetic_bscan(
    regime: RegimeConfig,
    rng: np.random.Generator,
    n_positions: int = 64,
    defects: list[DefectConfig] | None = None,
    beam_width_positions: int = 8,
    return_mask: bool = False,
    mask_threshold: float = 0.1,
    defect_prob: float = 0.5,
) -> BscanResult:
    """Generate a synthetic B-scan by stacking A-scans with spatial defect model.

    Args:
        regime: RegimeConfig for parameter sampling.
        rng: NumPy random generator.
        n_positions: Number of scan positions (rows in B-scan).
        defects: Explicit defect list with position_mm set. If None, samples 0 or 1
                 defect internally using defect_prob.
        beam_width_positions: Defect echo visible over this many positions.
                 NOTE: only the dominant defect (highest effective reflectivity)
                 is rendered per position. Extension 6 may need additive
                 rendering for overlapping defects.
        return_mask: When False, result.mask is None. When True, mask is computed
                     from echo thresholding in the noised domain (tracks jitter).
        mask_threshold: Fraction of peak echo amplitude for mask threshold.
        defect_prob: Probability of defect when defects=None. Use 1.0 to
                     guarantee a defect (used by generate_bscan_dataset to
                     honor flaw_ratio).

    Returns:
        BscanResult with bscan, label, and optional mask.
    """
    # Sample shared material parameters (constant across scan)
    # Use severity=0 for base params — defects are handled separately
    base_params = sample_trace_params(regime, rng, severity_class=0)

    # If defects not provided, sample 0 or 1
    if defects is None:
        has_defect = rng.random() < defect_prob
        if has_defect:
            severity = rng.choice([1, 2])
            defect_cfg = sample_defect(
                rng, severity, regime.defect_depth_mm,
                thickness_mm=base_params.thickness_mm,
            )
            # Place defect at a random position along the scan
            margin = beam_width_positions
            defect_cfg = DefectConfig(
                depth_mm=defect_cfg.depth_mm,
                reflectivity=defect_cfg.reflectivity,
                severity_label=defect_cfg.severity_label,
                position_mm=float(rng.integers(margin, max(margin + 1, n_positions - margin))),
            )
            defects = [defect_cfg]
        else:
            defects = []

    n_samples = base_params.n_samples
    bscan = np.zeros((n_positions, n_samples))
    mask = np.zeros((n_positions, n_samples), dtype=bool) if return_mask else None
    any_defect_rendered = False

    for pos in range(n_positions):
        params = _copy_trace_params(base_params)

        # Accumulate defect contributions for this position
        # Start with no defect in the trace
        params.has_defect = False
        params.defect_reflectivity = 0.0
        params.defect_depth_mm = 0.0

        # For each defect, compute beam sensitivity and find the dominant one
        # (the one with highest effective reflectivity at this position)
        best_sensitivity = 0.0
        best_defect = None
        for defect in defects:
            if defect.position_mm is None:
                continue
            sensitivity = _beam_sensitivity(pos, defect.position_mm, beam_width_positions)
            if sensitivity < 0.01:
                continue
            effective = defect.reflectivity * sensitivity
            if effective > best_sensitivity:
                best_sensitivity = effective
                best_defect = defect
                params.has_defect = True
                params.defect_depth_mm = defect.depth_mm
                params.defect_reflectivity = effective
                any_defect_rendered = True

        # Save rng state before noise so we can replay identical noise for mask
        if return_mask:
            noise_rng_state = rng.bit_generator.state

        # Generate clean trace and apply noise
        clean = generate_trace(params)
        noisy = apply_all_noise(clean, params, rng)
        bscan[pos] = noisy

        # Compute mask for this position if requested.
        # Mask is built in the noised domain so it tracks jitter/dropout.
        if return_mask and best_defect is not None:
            params_nodefect = _copy_trace_params(params)
            params_nodefect.has_defect = False
            params_nodefect.defect_reflectivity = 0.0
            params_nodefect.defect_depth_mm = 0.0
            clean_nodefect = generate_trace(params_nodefect)
            # Replay identical noise on no-defect trace
            mask_rng = np.random.default_rng(0)
            mask_rng.bit_generator.state = noise_rng_state
            noisy_nodefect = apply_all_noise(clean_nodefect, params_nodefect, mask_rng)
            defect_echo = noisy - noisy_nodefect
            peak = np.abs(defect_echo).max()
            if peak > 0:
                mask[pos] = np.abs(defect_echo) > mask_threshold * peak

    # Label based on whether any defect actually contributed to the B-scan
    label = 1 if any_defect_rendered else 0
    return BscanResult(bscan=bscan, label=label, mask=mask)


def generate_bscan_dataset(
    regime: RegimeConfig,
    n_samples: int,
    seed: int,
    flaw_ratio: float = 0.5,
    n_positions: int = 64,
) -> dict:
    """Generate a dataset of synthetic B-scans.

    Args:
        regime: RegimeConfig for parameter sampling.
        n_samples: Number of B-scans to generate.
        seed: Random seed.
        flaw_ratio: Fraction of B-scans with flaws.
        n_positions: Scan positions per B-scan.

    Returns:
        Dict with 'bscans' (N, n_positions, 1024) and 'labels' (N,).
    """
    rng = np.random.default_rng(seed)
    bscans = []
    labels = []

    for _ in range(n_samples):
        has_defect = rng.random() < flaw_ratio
        result = generate_synthetic_bscan(
            regime, rng, n_positions=n_positions,
            defects=None if has_defect else [],
            defect_prob=1.0,  # honor flaw_ratio — no second coin flip
        )
        bscans.append(result.bscan)
        labels.append(result.label)

    return {
        "bscans": np.array(bscans),
        "labels": np.array(labels),
    }
