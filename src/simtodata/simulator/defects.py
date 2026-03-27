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
    """Classify defect severity: 0=none, 1=low, 2=high.

    Uses HIGH_REFLECTIVITY_RANGE[0] as the threshold between low and high,
    so values in the gap (LOW upper, HIGH lower) are classified as low.
    """
    if reflectivity <= 0:
        return 0
    if reflectivity < HIGH_REFLECTIVITY_RANGE[0]:
        return 1
    return 2


def sample_defect(
    rng: np.random.Generator, severity: int, depth_range: tuple, thickness_mm: float | None = None
) -> DefectConfig:
    """Sample a defect configuration for a given severity class.

    Args:
        rng: NumPy random generator.
        severity: 0=none, 1=low, 2=high.
        depth_range: (min_mm, max_mm) for defect placement.
        thickness_mm: If provided, clamps max depth to thickness_mm - 1.0
                      so the defect echo always precedes the back-wall echo.
    """
    if severity == 0:
        return DefectConfig(depth_mm=0.0, reflectivity=0.0, severity_label=0)
    max_depth = depth_range[1]
    if thickness_mm is not None:
        max_depth = min(max_depth, thickness_mm - 1.0)
    if max_depth <= depth_range[0]:
        max_depth = depth_range[0] + 0.1
    depth = rng.uniform(depth_range[0], max_depth)
    if severity == 1:
        reflectivity = rng.uniform(*LOW_REFLECTIVITY_RANGE)
    else:
        reflectivity = rng.uniform(*HIGH_REFLECTIVITY_RANGE)
    return DefectConfig(depth_mm=depth, reflectivity=reflectivity, severity_label=severity)
