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
        defect_reflectivity=(0.05, 0.9),
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
                signals = torch.from_numpy(data["signals"]).unsqueeze(1)
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
