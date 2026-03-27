# sim-to-data

Synthetic ultrasonic inspection pipeline for evaluating defect detectors under controlled domain shift.

## Problem

Industrial ultrasonic inspection systems trained on one sensor/material configuration degrade when deployed on a different one. Real shifted data is scarce, expensive, and hard to label. This project uses a physics-based forward model to generate synthetic A-scan traces with configurable domain shift, then benchmarks how well standard detectors transfer across regimes.

## Approach

1. **Simulate**: A pulse-echo forward model generates 1D A-scan traces with surface echo, back-wall echo, and optional defect reflections. Configurable noise (Gaussian, baseline drift, gain variation, temporal jitter, masked dropout) simulates sensor degradation.
2. **Train**: A 1D CNN and non-neural baselines (logistic regression, gradient boosting) are trained on source-regime data.
3. **Shift**: A shifted regime widens material velocity, attenuation, frequency, and noise ranges to simulate deployment on a different sensor/material configuration.
4. **Adapt**: Fine-tuning on small labeled samples from the shifted regime measures adaptation efficiency.

## Benchmarks

| ID | Setup | Eval Set | Description |
|----|-------|----------|-------------|
| B0a | LogReg → Source features | Source test | Logistic regression baseline |
| B0b | GradBoost → Source features | Source test | Gradient boosting baseline |
| B0c | GradBoost → Source features | Shifted test | Baseline transfer gap |
| B1 | CNN trained on Source | Source test | Source performance ceiling |
| B2 | CNN trained on Source | Shifted test | Transfer gap (shift hurts) |
| B3 | CNN trained on Randomized | Shifted test | Domain randomization helps |
| B4 | B1 + fine-tune on Shifted | Shifted test | Fine-tuning recovers |
| B5 | B3 + fine-tune on Shifted | Shifted test | Best combined strategy |

**Expected pattern**: B1 > B2 (shift hurts), B3 > B2 (randomization helps), B4 > B2 (fine-tuning helps), B5 >= B3 and B4 (best combined).

## Additional Experiments

- **Robustness sweep**: Evaluates models across 5 shift intensity levels (none → extreme), with material and noise parameters progressively widening from source to shifted ranges.
- **Adaptation efficiency curve**: Measures macro-F1 vs fine-tune sample count (0, 25, 50, 100, 200) for source-pretrained and randomized-pretrained models.
- **Calibration analysis**: Reliability diagrams and ECE for all benchmarked models.

## Honest Scope

- All data is **purely synthetic** — the forward model is a simplified pulse-echo simulation, not a validated physics engine. Results demonstrate methodology, not field-ready performance.
- The "domain shift" is **controlled and parametric** — real-world shift involves corrosion, coupling variation, geometry changes, and other factors not modeled here.
- **No domain adaptation methods** (DANN, MMD, etc.) are implemented. The study compares naive transfer, domain randomization, and supervised fine-tuning only.
- The defect model is a **single point reflector** — real defects have complex geometries (cracks, porosity, delamination) that produce different echo patterns.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate data + run all experiments + produce figures
python experiments/run_all.py

# Quick mode (small datasets, reduced epochs)
python experiments/run_all.py --quick
```

## Makefile

```bash
make generate        # Generate synthetic datasets
make train-baselines # Run B0a-B0c baselines
make train-cnn       # Run B1-B5 CNN experiments
make evaluate        # Run robustness + adaptation sweeps
make figures         # Generate all figures
make all             # Full pipeline
make test            # Run test suite
make lint            # Ruff lint check
make clean           # Remove generated artifacts
```

## Engineering

- **112 tests** across 14 test files
- **CI**: GitHub Actions (lint + test on Python 3.10)
- **Reproducibility**: All experiment scripts seed PyTorch, NumPy, and DataLoader generators
- **Lint**: ruff, line-length 100

## Project Structure

```
sim-to-data/
├── src/simtodata/
│   ├── simulator/          # Forward model, defects, noise, regime config
│   ├── data/               # Dataset generation, PyTorch dataset, transforms
│   ├── features/           # Hand-crafted feature extraction
│   ├── models/             # CNN, baselines, training, prediction, factory
│   └── evaluation/         # Metrics, calibration, robustness, adaptation
├── configs/                # YAML configs for simulator and models
├── experiments/            # Experiment scripts and figure generation
├── tests/                  # Test suite
├── docs/plans/             # Design and implementation plans
└── Makefile
```
