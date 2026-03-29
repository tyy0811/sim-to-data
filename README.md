# sim-to-data

Synthetic ultrasonic inspection pipeline benchmarking defect detectors under controlled domain shift.

## Problem

Industrial ultrasonic inspection systems trained on one sensor/material configuration degrade when deployed on a different one. Real shifted data is scarce, expensive, and hard to label. This project uses a physics-based forward model to generate synthetic A-scan traces with configurable domain shift, then benchmarks how well standard detectors transfer across regimes.

## Approach

1. **Simulate**: A pulse-echo forward model generates 1D A-scan traces with surface echo, back-wall echo, and optional defect reflections. Configurable noise (Gaussian, baseline drift, gain variation, temporal jitter, masked dropout) simulates sensor degradation.
2. **Train**: A 1D CNN and non-neural baselines (logistic regression, gradient boosting) are trained on source-regime data.
3. **Shift**: A shifted regime widens material velocity, attenuation, frequency, and noise ranges to simulate deployment on a different sensor/material configuration.
4. **Adapt**: Fine-tuning on small labeled samples from the shifted regime measures adaptation efficiency.

### Example Traces

<p align="center">
  <img src="docs/figures/example_traces.png" width="800" alt="Example A-scan traces across source and shifted regimes for each defect severity class">
</p>

## Results

All CNN results (B1-B5) are reported as mean ± std across 5 training seeds on a fixed dataset. Baseline models (B0a-B0c) use deterministic sklearn estimators and produce identical results on the fixed dataset.

| ID | Setup | Eval Set | Macro-F1 | AUROC | ECE |
|----|-------|----------|----------|-------|-----|
| B0a | LogReg | Source test | 0.438 | 0.635 | 0.008 |
| B0b | GradBoost | Source test | 0.510 | 0.713 | 0.045 |
| B0c | GradBoost | Shifted test | 0.225 | 0.510 | 0.460 |
| B1 | CNN | Source test | **0.837 ± 0.006** | **0.951 ± 0.003** | 0.018 ± 0.006 |
| B2 | CNN | Shifted test | 0.265 ± 0.011 | 0.538 ± 0.005 | 0.609 ± 0.010 |
| B3 | CNN (randomized) | Shifted test | **0.542 ± 0.004** | **0.738 ± 0.006** | 0.098 ± 0.032 |
| B4 | B1 + fine-tune | Shifted test | 0.368 ± 0.008 | 0.546 ± 0.006 | 0.391 ± 0.032 |
| B5 | B3 + fine-tune | Shifted test | **0.550 ± 0.005** | **0.747 ± 0.006** | 0.071 ± 0.022 |

### Key Findings

Variance reflects training randomness (initialization, batch order) on a fixed dataset, not data-sampling variance.

- **Shift hurts consistently**: B1 → B2 shows a &Delta; = -0.57 F1 drop; all 5 B2 runs fall below all 5 B1 runs (no overlap in observed ranges).
- **Randomization helps reliably**: B3 (0.542 ± 0.004) vs B2 (0.265 ± 0.011) — a stable +0.28 improvement with low variance.
- **Fine-tuning preserves gains**: B5 (0.550 ± 0.005) &asymp; B3 (0.542 ± 0.004) — 200 adaptation samples do not degrade randomized performance.
- **CNN justified**: B1 (0.837 ± 0.006) >> B0b (0.510) on source.
- **Calibration**: B3/B5 have ECE &le; 0.10 while B2 has ECE = 0.61 — domain randomization produces better-calibrated models.

### Failure Mode Analysis

Under domain shift, the model collapses toward predicting high-severity for nearly all inputs. In B2, no-defect recall drops to 3%, low-severity to 18%, and high-severity inflates to 86% — the model predicts "high" regardless of true class. Domain randomization (B5) recovers no-defect recall to 75% and low-severity to 40%, though high-severity recall drops to 52% as the model distributes predictions more evenly.

Low-severity defects remain the hardest class across all conditions — the safety-critical failure pattern is that subtle defects are missed, not obvious ones.

<p align="center">
  <img src="docs/figures/confusion_matrices.png" width="800" alt="Confusion matrices for B1, B2, and B5 showing per-class recall under domain shift">
  <br><em>Representative seed (seed=42). Per-class recalls are consistent across all 5 training seeds.</em>
</p>

### Calibration

<p align="center">
  <img src="docs/figures/calibration_diagram.png" width="700" alt="Reliability diagrams comparing B2 and B5 calibration">
  <br><em>Representative seed (seed=42).</em>
</p>

B2 (source-only, shifted evaluation) is severely miscalibrated (ECE = 0.609 ± 0.010) — the model is overconfident on incorrect predictions. B5 (randomized + fine-tuned) achieves ECE = 0.071 ± 0.022, an 8.6&times; reduction in calibration error, indicating that domain randomization improves not just accuracy but also prediction trustworthiness.

### Robustness Under Increasing Shift

Robustness and adaptation results below are from a single representative seed (seed=42). Each intensity level generates a fresh test set with progressively wider parameter ranges.

<p align="center">
  <img src="docs/figures/robustness_curve.png" width="600" alt="F1 vs shift intensity for GradBoost, source CNN, and randomized+finetuned CNN">
</p>

| Intensity | GradBoost | CNN (source) | CNN (rand+ft) |
|-----------|-----------|--------------|---------------|
| none | 0.510 | 0.887 | 0.721 |
| low | 0.326 | 0.398 | 0.683 |
| medium | 0.202 | 0.252 | 0.496 |
| high | 0.197 | 0.181 | 0.459 |
| extreme | 0.159 | 0.160 | 0.405 |

### Adaptation Efficiency

<p align="center">
  <img src="docs/figures/adaptation_curve.png" width="600" alt="F1 vs fine-tune sample count for source-pretrained and randomized-pretrained models">
</p>

| Samples | Source-pretrained F1 | Randomized-pretrained F1 |
|---------|---------------------|-------------------------|
| 0 | 0.288 | 0.544 |
| 25 | 0.362 | 0.536 |
| 50 | 0.371 | 0.545 |
| 100 | 0.362 | 0.539 |
| 200 | 0.353 | 0.547 |

Fine-tuning the source-pretrained model plateaus at F1 &asymp; 0.37 regardless of sample count (25-200), while the randomized model starts at 0.54 — domain randomization cannot be replaced by more target labels at this scale.

## Statistical Methodology

CNN experiments (B1-B5) are repeated across 5 random seeds controlling model initialization and training order on a fixed dataset (seed=42). Results are reported as mean ± standard deviation. Baseline models (B0a-B0c) use deterministic sklearn estimators and produce identical results on the fixed dataset.

With 5 seeds, formal significance testing has limited statistical power. We report effect sizes and consistency across runs rather than p-values. All 5 B3 runs individually outperform all 5 B2 runs (no overlap), providing strong qualitative evidence for the randomization effect.

## Context

Domain shift in sensor-based ML is studied in sim-to-real robotics (Tobin et al., 2017), medical imaging (Stacke et al., 2020), and theoretically via domain divergence bounds (Ben-David et al., 2010). This project applies domain randomization and supervised fine-tuning to synthetic ultrasonic inspection, testing whether these simple strategies suffice before reaching for more complex domain adaptation machinery (Ganin et al., 2016; Sun & Saenko, 2016).

## Sim-to-Real Transfer (B-Scan)

To test whether the synthetic pipeline transfers to real measurements,
we adapt the 1D simulator to generate synthetic B-scans (stacking
adjacent A-scans with a spatial defect model) and evaluate on real
phased-array weld inspection data from Virkkunen et al. (2021).

### Modality Gap

The synthetic and real data come from fundamentally different setups:

| | Synthetic | Real (Virkkunen) |
|---|-----------|-----------------|
| Mode | Longitudinal pulse-echo | TRS phased array (shear) |
| Material | Generic metal | Austenitic 316L stainless steel weld |
| Frequency | 1.5-7.0 MHz | 1.8 MHz |
| Defect model | Point reflector | Thermal fatigue cracks |
| Noise sources | Gaussian, drift, jitter, dropout | Grain noise, mode conversion |

This is a **severe out-of-distribution stress test**, not a matched
sim-to-real validation.

<p align="center">
  <img src="docs/figures/sim_vs_real_bscans.png" width="800"
       alt="Synthetic vs real B-scan comparison">
</p>

### Results

| Experiment | Train | Eval | F1 | AUROC |
|-----------|-------|------|----|-------|
| SB1 | Synthetic source | Synthetic source | 0.834 | 0.923 |
| SB2 | Synthetic source | Synthetic shifted | 0.677 | 0.496 |
| SB3 | Synthetic randomized | Synthetic shifted | 0.532 | 0.487 |
| SR1 | Synthetic source | **Real** | 0.714 | 0.500 |
| SR2 | Synthetic randomized | **Real** | 0.714 | 0.176 |

<p align="center">
  <img src="docs/figures/sim_to_real_results.png" width="600"
       alt="Sim-to-real transfer results">
</p>

### Interpretation

SB1 confirms the 2D CNN learns source-regime B-scans well (AUROC = 0.923). On shifted and real data, AUROC drops to ~0.5 (random) or below, meaning the models lose all discriminative ability. The inflated F1 values (0.68-0.71) reflect class imbalance: predicting all-flaw on a 55% flaw dataset yields F1 ~0.71 without any real discrimination.

Domain randomization (SB3 vs SB2) does not rescue B-scan transfer in this setup, unlike the 1D A-scan results where it provided a +0.28 F1 gain. This likely reflects the 2D spatial structure: the Gaussian beam profile creates spatially coherent defect patterns that are easier for the CNN to memorize, but that spatial structure does not transfer across regimes.

The real-data results (SR1/SR2 AUROC &le; 0.50) confirm the expected outcome: a simplified pulse-echo simulator cannot produce features that generalize to real TRS phased-array weld inspection. Bridging this gap would require physics-informed simulation of shear-wave propagation, realistic grain-noise models, and calibrated transducer beam profiles — none of which are in scope for this project.

## Honest Scope

- Primary experiments use **synthetic data**. A sim-to-real stress test against real phased-array weld data (Virkkunen et al., 2021) is included. The extreme modality gap between the simplified pulse-echo simulator and real TRS phased-array inspection means this tests the limits of synthetic training, not its production readiness.
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

# Multi-seed benchmark (5 training seeds)
python experiments/run_multiseed.py
python experiments/aggregate_multiseed.py
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

- **152 tests** across 20 test files
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
└── Makefile
```
