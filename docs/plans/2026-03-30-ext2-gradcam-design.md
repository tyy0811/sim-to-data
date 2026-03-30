# Extension 2: Grad-CAM Interpretability Design

## Goal

Show what the 1D CNN attends to and how attention shifts under domain shift. Two figures that complement the existing confusion matrix and calibration analysis.

## Approach

Vanilla Grad-CAM implemented from scratch (~20 lines per function, no `captum` dependency). Hook-based: register forward/backward hooks on the target Conv1d layer, compute channel-weighted activation map, ReLU, interpolate to input size.

## Module: `src/simtodata/evaluation/interpretability.py`

Three stateless functions:

**`gradcam_1d(model, x, target_class, target_layer) -> np.ndarray`**
- Registers forward hook (save activations) and backward hook (save gradients) on `target_layer`
- Forward pass, backward pass on `output[0, target_class]`
- Weights = global avg pool of gradients over time dim
- CAM = ReLU(sum of weighted activations), interpolated to input length
- Hooks removed in `try/finally` to prevent leaking
- Returns (1024,) numpy array

**`gradcam_2d(model, x, target_class, target_layer) -> np.ndarray`**
- Identical logic, `mean(dim=(2,3))` for spatial pooling, `mode='bilinear'` for interpolation
- Included for future use (Ext 6, or if B-scan models improve). Not used in this extension's figures.

**`compute_attribution_batch(model, dataset, target_layer, n_samples=50, seed=42) -> dict`**
- Loops over `n_samples` from `dataset` (seeded selection)
- Calls `gradcam_1d` for the model's predicted class on each sample
- Normalizes each attribution map to [0, 1] before collecting (per-sample normalization so no single sample dominates the average)
- Returns `{"signals": np.ndarray, "attributions": np.ndarray, "labels": np.ndarray, "predictions": np.ndarray}`

## Tests: `tests/test_interpretability.py`

7 tests, all using `DefectCNN1D` with random weights (no checkpoints needed):

| Test | Property |
|------|----------|
| `test_output_shape` | Returns (1024,) for 1024-sample input |
| `test_output_finite` | No NaN/Inf |
| `test_output_nonnegative` | ReLU guarantees >= 0 |
| `test_different_classes_differ` | Class 0 != class 1 attributions |
| `test_hooks_cleaned_up` | No forward/backward hooks remain after call |
| `test_batch_returns_expected_keys` | Dict has signals, attributions, labels, predictions |
| `test_batch_attributions_normalized` | Each map in [0, 1], max == 1.0 |

## Figure A: Attribution comparison grid

`docs/figures/gradcam_grid.png` — 2x3 grid.

```
              B1 (source eval)     B2 (shifted eval)     B5 (shifted eval)
flaw (high):  [signal + heatmap]   [signal + heatmap]    [signal + heatmap]
no-flaw:      [signal + heatmap]   [signal + heatmap]    [signal + heatmap]
```

Each cell: raw signal in gray, Grad-CAM as semi-transparent colored fill overlaid.

Models loaded: B1 (`B1_cnn1d_source.pt`) and B5 (`B5_cnn1d_randomized_finetuned.pt`). B2 is B1 evaluated on shifted test data (same checkpoint, different input).

Sample selection (seed=42, deterministic):
- **Flaw row**: First high-severity (severity=2) sample. For B2 column, pick a *correctly classified* high-severity sample — B2 has 86% recall on high-severity, so this shows the "right answer, wrong reason" pattern where Grad-CAM reveals the model attends to noise/backwall rather than the defect echo.
- **No-flaw row**: First severity=0 sample.

## Figure B: Mean attribution profile

`docs/figures/gradcam_mean_profile.png` — single plot.

Three lines over a gray signal envelope:
- **Blue**: B1 on source test (50 high-severity flaw samples)
- **Orange**: B2 on shifted test (same 50-sample selection logic)
- **Green**: B5 on shifted test (same)

Gray background: mean |signal| envelope from the 50 **source** samples only (fixed spatial context, not recomputed per condition).

Computation: `gradcam_1d` per sample for predicted class, normalize to [0, 1] per sample, then average across samples.

## Experiment script: `experiments/generate_gradcam_figures.py`

Loads models (B1, B5), generates source/shifted test datasets, computes attributions, produces both figures. CLI args: `--config`, `--models-dir`, `--output-dir`, `--n-samples`, `--seed`.

## README addition

Short section after "Failure Mode Analysis", before "Calibration":

```markdown
### Attribution Analysis (Grad-CAM)

[2-3 sentences + grid figure + profile figure + honest caveat about
qualitative diagnostics]
```

Honest framing: "Qualitative diagnostics showing where the model attends, not proof of learned physical reasoning."
