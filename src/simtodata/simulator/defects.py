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
