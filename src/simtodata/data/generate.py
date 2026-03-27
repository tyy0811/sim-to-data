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
