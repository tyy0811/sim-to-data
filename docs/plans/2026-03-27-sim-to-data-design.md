# sim-to-data Design Document

**Project:** Synthetic inspection signals for defect detection under domain shift
**Author:** Jane Yeung
**Date:** 2026-03-27
**Hardware constraint:** CPU-only (i7 quad-core, 16GB RAM)
**Estimated scope:** 9 working days + 1 buffer day

---

## 1. One-Sentence Story

> Built a synthetic ultrasonic inspection pipeline to train and evaluate defect detectors under sensor and material domain shift.

---

## 2. Scope

This project studies **synthetic-to-shifted-synthetic transfer** as a controlled proxy for scarce-label inspection deployment. It does not claim direct real-world performance.

Core question: If you train on clean simulator output and deploy under degraded sensor conditions, how much performance do you lose — and how cheaply can you recover it?

**Good framing:** simulation-to-deployment-shift benchmark, synthetic-to-shifted-synthetic transfer study, proxy for scarce-label inspection settings, controlled domain shift analysis for engineering ML.

**Bad framing:** real-world transfer results, digital twin, production inspection system, state-of-the-art domain adaptation.

---

## 3. Physical Forward Model

### Pulse-echo A-scan

A transducer emits a pulse into a material, the pulse reflects off internal boundaries (defects), and the transducer records the returning signal as a 1D time series.

**Mathematical model:**

```
s(t) = A_surface * p(t - t_surface)
     + A_defect  * p(t - t_defect) * H(defect_present)
     + A_backwall * p(t - t_backwall)
     + noise(t)
```

Where:
- `p(t)` is a Gabor wavelet: `exp(-((t-t0)^2 / (2*sigma^2))) * sin(2*pi*f_center*(t-t0))`
- `t_defect = 2 * d_defect / v_material` (two-way travel time)
- `t_backwall = 2 * d_thickness / v_material`
- `A_defect = reflectivity * exp(-alpha * 2 * d_defect)`
- `A_backwall = A_surface * exp(-alpha * 2 * d_thickness)`
- **`A_surface` is fixed at 1.0; all other amplitudes are relative to it.** Surface coupling is held constant to isolate material and sensor degradation effects.

### Signal dimensions

- Sampling rate: 50 MHz, 1024 time samples per trace
- Shape: `(1024,)` per trace

### Labels — three-class severity

| Class | Label | Description |
|-------|-------|-------------|
| 0 | no_defect | Only surface + back-wall echoes |
| 1 | low_severity | Defect reflectivity 0.1-0.3 |
| 2 | high_severity | Defect reflectivity 0.4-0.8 |

These are simulator-derived severity classes based on reflectivity thresholds, not physical defect size measurements.

---

## 4. Three Regimes

### Source (clean synthetic)
- Narrow parameter ranges, high SNR (20-40 dB), no degradation
- Purpose: Training data. Upper-bound performance.

### Shifted (degraded synthetic)
- Wider parameter ranges, lower SNR (5-40 dB), baseline drift, gain variation, jitter, masked dropout
- Purpose: Simulates deployment-like conditions. Measures transfer gap.

### Randomized (full-range training)
- Same parameter ranges as shifted
- Purpose: Domain randomization — training on the full parameter range rather than the narrow source subset, not a separate distribution
- Note: uniform sampling over full ranges. If B3 shows degraded source performance vs B1, consider 50/50 mixture of source/shifted sampling. This is a one-line tunable in `regime.py`.

### Dataset sizes

| Split | Regime | Size |
|-------|--------|------|
| train | source | 20,000 |
| val | source | 2,000 |
| test_source | source | 3,000 |
| test_shifted | shifted | 3,000 |
| train_adapt | shifted | 200 |
| train_randomized | randomized | 20,000 |

Total: ~48,200 samples, ~190 MB.

---

## 5. Parameter Table

| Parameter | Symbol | Range (source) | Range (shifted) | Unit |
|-----------|--------|----------------|-----------------|------|
| Material thickness | d_thick | 10-30 | 10-30 | mm |
| Material velocity | v_mat | 5800-6200 | 5500-6500 | m/s |
| Attenuation coeff | alpha | 0.01-0.05 | 0.01-0.10 | Np/mm |
| Center frequency | f_c | 2.0-5.0 | 1.5-7.0 | MHz |
| Pulse bandwidth | sigma | 0.5-1.5 | 0.3-2.0 | us |
| Defect depth | d_defect | 2-28 | 2-28 | mm |
| Defect reflectivity | A_defect | 0.1-0.8 | 0.05-0.9 | - |
| SNR | - | 20-40 | 5-40 | dB |
| Baseline drift | - | 0 | 0-0.3 | - |
| Sensor gain variation | - | 1.0 | 0.7-1.3 | - |
| Temporal jitter | - | 0 | 0-2 | samples |
| Masked dropout | - | 0 gaps | 0-3 gaps, 5-20 samples each | - |

---

## 6. Models

### Non-neural baselines (B0)

11 hand-crafted features per trace:
- `n_peaks`: peak count above adaptive threshold
- `peak_amplitudes_top3` (3 features): amplitudes of 3 largest peaks
- `peak_times_top3` (3 features): time indices of 3 largest peaks
- `signal_energy`: sum of squared amplitudes
- `spectral_centroid`: weighted mean of FFT frequency bins
- `mid_region_energy`: energy between 5%-85% of trace (fixed window, no physics params needed)
- `inter_peak_ratio`: ratio of 2nd-largest to largest peak amplitude

Classifiers: LogisticRegression (with StandardScaler), GradientBoostingClassifier (100 trees).

### 1D CNN (primary, Model A)

```
Input: (batch, 1, 1024)
  -> Conv1d(1, 32, kernel=7, stride=2) -> BN -> ReLU -> MaxPool(2)
  -> Conv1d(32, 64, kernel=5, stride=1) -> BN -> ReLU -> MaxPool(2)
  -> Conv1d(64, 128, kernel=3, stride=1) -> BN -> ReLU -> AdaptiveAvgPool(1)
  -> Flatten -> Linear(128, 64) -> ReLU -> Dropout(0.3)
  -> Linear(64, 3)
```

~100K parameters. Adam lr=1e-3, ReduceLROnPlateau, early stopping.

### Spectrogram 2D CNN (comparison, Model B — cuttable)

STFT (window=64, hop=16) -> log magnitude spectrogram -> (1, 33, 61).
Three Conv2d layers, ~50K parameters. Same training regime.

---

## 7. Benchmark Experiments

| ID | Model | Training | Eval | Purpose |
|----|-------|----------|------|---------|
| B0a | LogReg | Source features | Source | Non-neural baseline |
| B0b | GradBoost | Source features | Source | Non-neural baseline |
| B0c | GradBoost | Source features | Shifted | Baseline transfer gap |
| B1 | 1D CNN | Source (20K) | Source | CNN upper bound |
| B2 | 1D CNN | Source (20K) | Shifted | Transfer gap |
| B3 | 1D CNN | Randomized (20K) | Shifted | Randomization helps? |
| B4 | 1D CNN | Source + fine-tune 200 | Shifted | Fine-tuning helps? |
| B5 | 1D CNN | Rand + fine-tune 200 | Shifted | Best combined? |
| B6* | Spec CNN | Source (20K) | Source | Spectrogram baseline |
| B7* | Spec CNN | Source (20K) | Shifted | Spectrogram transfer |
| B8* | Spec CNN | Rand + fine-tune 200 | Shifted | Spectrogram adapted |

*B6-B8 only if Day 6 executed.

Expected pattern: B1 > B2 (shift hurts), B3 > B2 (randomization helps), B4 > B2 (fine-tuning helps), B5 >= B3 and B4 (best combined).

---

## 8. Key Implementation Notes

### Temporal jitter: zero-padded, not circular
Circular shift would wrap back-wall echoes to trace start. Zero-pad left/right instead.

### Masked dropout: contiguous windows, not random points
Models coupling loss or acquisition gaps. Physically plausible.

### Mid-region energy: fixed percentage window
Use 5%-85% of trace. Baselines should not depend on physics parameters.

### A_surface = 1.0
Fixed. Everything else relative. Document in forward model docstring.

### Randomized = shifted ranges
Intentional. Difference is source training never touches degradation ranges. Note as tunable in regime.py. Only add mixture weighting if B3 degrades source performance vs B1.

---

## 9. Repo Structure

```
sim-to-data/
  src/simtodata/
    simulator/   (forward_model, defects, noise, regime)
    data/        (generate, dataset, transforms)
    features/    (extract)
    models/      (baselines, cnn1d, cnn2d_spectrogram, factory, train, predict)
    evaluation/  (metrics, calibration, robustness, adaptation_curve, benchmark)
  configs/       (simulator.yaml, model configs, experiment configs)
  experiments/   (run scripts, generate_figures)
  tests/         (16+ test files)
  docs/          (architecture diagram, figures)
```

---

## 10. Cut Priority

| Priority | Component | Day | Cut? |
|----------|-----------|-----|------|
| 1-7 | Core pipeline + README | 1-5,9 | Never |
| 8 | Robustness sweep | 5 | Reduce to 3 intensities |
| 9 | Tests >= 50 + CI | 8 | Cut to 35 + smoke |
| 10 | Calibration diagram | 5 | Cut |
| 11 | Spectrogram B6-B8 | 6 | Cut entire day |
| 12-16 | Adaptation curve, confusion, Makefile, traces, buffer | 7-10 | Cut |

5-day MVP = priorities 1-7. 7-day strong = 1-10. 9-day full = 1-15.

---

## 11. Risk Mitigations

- Source accuracy >99%: reduce source SNR or tighten severity thresholds
- No transfer gap: widen shifted ranges until 10-30% gap
- Non-neural beats CNN: report honestly, check for bugs
- Scope creep: NO DANN, NO MMD, NO adversarial. Randomization + fine-tuning only.

---

## 12. Amendments (2026-03-27)

1. **Randomized = shifted ranges confirmed intentional.** Difference is training coverage, not distribution shape. Uniform sampling default; add mixture weighting only if B3 degrades source performance. Comment in regime.py.
2. **mid_region_energy added to feature extraction.** Fixed 5%-85% window. 11 total features.
3. **A_surface = 1.0 fixed.** All amplitudes relative. No surface coupling variation in v1. Document in docstring.
