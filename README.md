# sim-to-data

Synthetic ultrasonic inspection pipeline for evaluating defect detectors under controlled domain shift.

## Problem

Industrial ultrasonic inspection systems trained on one sensor/material configuration degrade when deployed on a different one. Real shifted data is scarce, expensive, and hard to label. This project uses a physics-based forward model to generate synthetic A-scan traces with configurable domain shift, then benchmarks how well standard detectors transfer across regimes.

## Approach

1. **Simulate**: A pulse-echo forward model generates 1D A-scan traces with surface echo, back-wall echo, and optional defect reflections. Configurable noise (Gaussian, baseline drift, gain variation, temporal jitter, masked dropout) simulates sensor degradation.
2. **Train**: A 1D CNN and non-neural baselines (logistic regression, gradient boosting) are trained on source-regime data.
3. **Shift**: A shifted regime widens material velocity, attenuation, frequency, and noise ranges to simulate deployment on a different sensor/material configuration.
4. **Adapt**: Fine-tuning on small labeled samples from the shifted regime measures adaptation efficiency.

## Results

| ID | Setup | Eval Set | Macro-F1 | AUROC | ECE |
|----|-------|----------|----------|-------|-----|
| B0a | LogReg | Source test | 0.438 | 0.635 | 0.008 |
| B0b | GradBoost | Source test | 0.510 | 0.713 | 0.045 |
| B0c | GradBoost | Shifted test | 0.225 | 0.510 | 0.460 |
| B1 | CNN → Source | Source test | **0.822** | **0.946** | 0.024 |
| B2 | CNN → Source | Shifted test | 0.288 | 0.543 | 0.575 |
| B3 | CNN → Randomized | Shifted test | **0.544** | **0.743** | 0.096 |
| B4 | B1 + fine-tune | Shifted test | 0.366 | 0.550 | 0.374 |
| B5 | B3 + fine-tune | Shifted test | **0.542** | **0.742** | 0.094 |

### Key Findings

- **Shift hurts**: B1 (0.82) → B2 (0.29) — a 65% F1 drop when deploying to shifted regime
- **Randomization helps**: B3 (0.54) vs B2 (0.29) — training on wider parameter ranges nearly doubles shifted-domain F1
- **Fine-tuning recovers partially**: B4 (0.37) > B2 (0.29) — 200 labeled shifted samples improve transfer
- **Best combined**: B5 (0.54) ≈ B3 (0.54) — fine-tuning preserves randomized performance
- **CNN justified**: B1 (0.82) >> B0b (0.51) on source; shift affects all models (B0c ~ B2)
- **Calibration**: B3/B5 have ECE ~ 0.09 while B2 has ECE = 0.58 — randomization produces better-calibrated models

### Robustness Under Increasing Shift

| Intensity | GradBoost | CNN (source) | CNN (rand+ft) |
|-----------|-----------|--------------|---------------|
| none | 0.510 | 0.887 | 0.721 |
| low | 0.326 | 0.398 | 0.683 |
| medium | 0.202 | 0.252 | 0.496 |
| high | 0.197 | 0.181 | 0.459 |
| extreme | 0.159 | 0.160 | 0.405 |

The randomized+finetuned CNN (B5) degrades much more gracefully than source-only models under increasing domain shift. At extreme intensity, B5 maintains F1 = 0.41 while source-only models collapse to ~0.16.

### Adaptation Efficiency

| Samples | Source-pretrained F1 | Randomized-pretrained F1 |
|---------|---------------------|-------------------------|
| 0 | 0.288 | 0.544 |
| 25 | 0.362 | 0.536 |
| 50 | 0.371 | 0.545 |
| 100 | 0.362 | 0.539 |
| 200 | 0.353 | 0.547 |

Domain randomization provides a substantially better starting point (0.54 vs 0.29 at 0 samples), reducing the need for labeled target data.

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
