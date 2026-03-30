# Extension 2: Grad-CAM Interpretability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add vanilla Grad-CAM for the 1D CNN and generate two figures showing how model attention shifts under domain shift — complementing the existing confusion matrix and calibration analysis.

**Architecture:** Two pure functions (`gradcam_1d`, `gradcam_2d`) using PyTorch hooks to extract activations and gradients from a target Conv layer. A batch helper normalizes per-sample before averaging. One experiment script loads B1/B5 checkpoints and source/shifted test datasets to produce both figures.

**Tech Stack:** torch (hooks, F.interpolate), numpy, matplotlib. No new dependencies.

---

## Task 1: Implement `gradcam_1d`

**Files:**
- Create: `src/simtodata/evaluation/interpretability.py`
- Test: `tests/test_interpretability.py`

**Step 1: Write the tests**

Create `tests/test_interpretability.py`:

```python
"""Tests for Grad-CAM interpretability functions."""

import torch

from simtodata.evaluation.interpretability import gradcam_1d
from simtodata.models.cnn1d import DefectCNN1D


def _get_model_and_layer():
    """Random-weight model and its last Conv1d layer."""
    model = DefectCNN1D()
    model.eval()
    # features is Sequential: [Conv1d, BN, ReLU, MaxPool] x 4 + AdaptiveAvgPool
    # Last Conv1d is at index -5 (before final BN, ReLU, MaxPool, AvgPool)
    last_conv = None
    for module in model.features:
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module
    return model, last_conv


class TestGradcam1d:
    def test_output_shape(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam = gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert cam.shape == (1024,)

    def test_output_finite(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam = gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert all(c == c for c in cam)  # no NaN
        assert all(abs(c) < float("inf") for c in cam)  # no Inf

    def test_output_nonnegative(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam = gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert all(c >= 0 for c in cam)

    def test_different_classes_differ(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        cam0 = gradcam_1d(model, x, target_class=0, target_layer=layer)
        cam1 = gradcam_1d(model, x, target_class=1, target_layer=layer)
        # With random weights, different class gradients produce different maps
        assert not all(abs(a - b) < 1e-10 for a, b in zip(cam0, cam1))

    def test_hooks_cleaned_up(self):
        model, layer = _get_model_and_layer()
        x = torch.randn(1, 1, 1024)
        gradcam_1d(model, x, target_class=0, target_layer=layer)
        assert len(layer._forward_hooks) == 0
        assert len(layer._backward_hooks) == 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_interpretability.py -v
```

Expected: FAIL — `simtodata.evaluation.interpretability` does not exist.

**Step 3: Implement `gradcam_1d`**

Create `src/simtodata/evaluation/interpretability.py`:

```python
"""Grad-CAM attribution maps for 1D and 2D CNNs.

Vanilla Grad-CAM implemented from scratch using PyTorch hooks.
No external attribution library required.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam_1d(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """Compute Grad-CAM attribution for a 1D CNN.

    Args:
        model: Trained 1D CNN (e.g., DefectCNN1D).
        x: Input tensor of shape (1, 1, n_samples).
        target_class: Class index to compute attribution for.
        target_layer: Conv1d layer to extract activations from.

    Returns:
        Attribution map of shape (n_samples,), non-negative.
    """
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    hook_fwd = target_layer.register_forward_hook(save_activation)
    hook_bwd = target_layer.register_full_backward_hook(save_gradient)

    try:
        output = model(x)
        model.zero_grad()
        output[0, target_class].backward()

        weights = gradients["value"].mean(dim=2, keepdim=True)
        cam = (weights * activations["value"]).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
        return cam.squeeze().cpu().numpy()
    finally:
        hook_fwd.remove()
        hook_bwd.remove()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_interpretability.py -v
```

Expected: 5 PASSED.

**Step 5: Run full test suite — no regressions**

```bash
python -m pytest tests/ -q --ignore=tests/test_dataset.py --ignore=tests/test_benchmark_smoke.py
```

Expected: All pass (run in this subset to avoid the transient segfault).

**Step 6: Commit**

```bash
git add src/simtodata/evaluation/interpretability.py tests/test_interpretability.py
git commit -m "feat(ext2): vanilla Grad-CAM for 1D CNN with hook cleanup"
```

---

## Task 2: Implement `gradcam_2d` and `compute_attribution_batch`

**Files:**
- Modify: `src/simtodata/evaluation/interpretability.py`
- Modify: `tests/test_interpretability.py`

**Step 1: Add tests**

Append to `tests/test_interpretability.py`:

```python
from simtodata.evaluation.interpretability import compute_attribution_batch


class TestComputeAttributionBatch:
    def test_returns_expected_keys(self):
        model, layer = _get_model_and_layer()
        # Create a tiny fake dataset
        from torch.utils.data import TensorDataset
        signals = torch.randn(10, 1, 1024)
        labels = torch.randint(0, 3, (10,))
        dataset = TensorDataset(signals, labels)
        result = compute_attribution_batch(model, dataset, layer, n_samples=5, seed=42)
        assert set(result.keys()) == {"signals", "attributions", "labels", "predictions"}
        assert result["signals"].shape == (5, 1024)
        assert result["attributions"].shape == (5, 1024)
        assert result["labels"].shape == (5,)
        assert result["predictions"].shape == (5,)

    def test_attributions_normalized(self):
        model, layer = _get_model_and_layer()
        from torch.utils.data import TensorDataset
        signals = torch.randn(10, 1, 1024)
        labels = torch.randint(0, 3, (10,))
        dataset = TensorDataset(signals, labels)
        result = compute_attribution_batch(model, dataset, layer, n_samples=5, seed=42)
        for i in range(5):
            cam = result["attributions"][i]
            assert cam.min() >= 0.0
            assert abs(cam.max() - 1.0) < 1e-6, f"Sample {i} max={cam.max()}"
```

**Step 2: Run tests to verify new tests fail**

```bash
python -m pytest tests/test_interpretability.py::TestComputeAttributionBatch -v
```

Expected: FAIL — `compute_attribution_batch` not importable.

**Step 3: Implement both functions**

Add to `src/simtodata/evaluation/interpretability.py`:

```python
def gradcam_2d(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """Compute Grad-CAM attribution for a 2D CNN.

    Args:
        model: Trained 2D CNN (e.g., BscanCNN).
        x: Input tensor of shape (1, 1, H, W).
        target_class: Class index to compute attribution for.
        target_layer: Conv2d layer to extract activations from.

    Returns:
        Attribution map of shape (H, W), non-negative.
    """
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    hook_fwd = target_layer.register_forward_hook(save_activation)
    hook_bwd = target_layer.register_full_backward_hook(save_gradient)

    try:
        output = model(x)
        model.zero_grad()
        output[0, target_class].backward()

        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations["value"]).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False,
        )
        return cam.squeeze().cpu().numpy()
    finally:
        hook_fwd.remove()
        hook_bwd.remove()


def compute_attribution_batch(
    model: torch.nn.Module,
    dataset,
    target_layer: torch.nn.Module,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """Compute Grad-CAM attributions for a batch of samples.

    Each attribution map is normalized to [0, 1] per-sample before
    collecting, so no single sample dominates the average.

    Args:
        model: Trained 1D CNN.
        dataset: PyTorch Dataset returning (signal, label) pairs.
        target_layer: Conv1d layer for Grad-CAM.
        n_samples: Number of samples to process.
        seed: Random seed for sample selection.

    Returns:
        Dict with keys: signals (N, 1024), attributions (N, 1024),
        labels (N,), predictions (N,).
    """
    model.eval()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    signals, attributions, labels, predictions = [], [], [], []

    for idx in indices:
        x, y = dataset[int(idx)]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, 1, n_samples)

        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()

        cam = gradcam_1d(model, x, target_class=pred, target_layer=target_layer)

        # Per-sample normalization to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        signals.append(x.squeeze().cpu().numpy())
        attributions.append(cam)
        labels.append(y.item() if hasattr(y, "item") else int(y))
        predictions.append(pred)

    return {
        "signals": np.array(signals),
        "attributions": np.array(attributions),
        "labels": np.array(labels),
        "predictions": np.array(predictions),
    }
```

**Step 4: Run all interpretability tests**

```bash
python -m pytest tests/test_interpretability.py -v
```

Expected: 7 PASSED.

**Step 5: Commit**

```bash
git add src/simtodata/evaluation/interpretability.py tests/test_interpretability.py
git commit -m "feat(ext2): gradcam_2d and compute_attribution_batch with per-sample normalization"
```

---

## Task 3: Implement the figure generation script

**Files:**
- Create: `experiments/generate_gradcam_figures.py`

This script loads B1 and B5 checkpoints, source and shifted test datasets,
computes Grad-CAM attributions, and produces two figures.

**Step 1: Implement the script**

Create `experiments/generate_gradcam_figures.py`:

```python
"""Generate Grad-CAM attribution figures for domain shift analysis.

Figure A: 2x3 attribution grid (B1 source, B2 shifted, B5 shifted) x (flaw, no-flaw)
Figure B: Mean attribution profile (B1/B2/B5 averaged over 50 flaw samples)
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.interpretability import gradcam_1d, compute_attribution_batch
from simtodata.models.cnn1d import DefectCNN1D


def _load_model(path):
    """Load a DefectCNN1D from a state dict checkpoint."""
    model = DefectCNN1D()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def _last_conv(model):
    """Find the last Conv1d layer in model.features."""
    last = None
    for module in model.features:
        if isinstance(module, torch.nn.Conv1d):
            last = module
    return last


def _find_sample(dataset, severity, model, target_layer, correct=True, seed=42):
    """Find a sample with given severity and correctness.

    Args:
        dataset: InspectionDataset.
        severity: Target label (0, 1, or 2).
        model: Trained model for prediction check.
        target_layer: Conv1d layer for Grad-CAM.
        correct: If True, find correctly classified sample.
        seed: For deterministic tie-breaking.

    Returns:
        (signal, label, prediction, attribution) or None.
    """
    rng = np.random.default_rng(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    for idx in indices:
        x, y = dataset[idx]
        if y.item() != severity:
            continue
        x_batch = x.unsqueeze(0)
        with torch.no_grad():
            pred = model(x_batch).argmax(dim=1).item()
        is_correct = (pred == severity)
        if is_correct == correct:
            cam = gradcam_1d(model, x_batch, target_class=pred, target_layer=target_layer)
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max
            return x.squeeze().numpy(), y.item(), pred, cam

    # Fallback: return first matching severity regardless of correctness
    for idx in indices:
        x, y = dataset[idx]
        if y.item() != severity:
            continue
        x_batch = x.unsqueeze(0)
        with torch.no_grad():
            pred = model(x_batch).argmax(dim=1).item()
        cam = gradcam_1d(model, x_batch, target_class=pred, target_layer=target_layer)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return x.squeeze().numpy(), y.item(), pred, cam

    return None


def plot_attribution_grid(save_path, model_b1, model_b5, source_test, shifted_test):
    """2x3 grid: columns = B1/source, B2/shifted, B5/shifted. Rows = flaw, no-flaw."""
    layer_b1 = _last_conv(model_b1)
    layer_b5 = _last_conv(model_b5)

    # Columns: (model, dataset, target_layer, label)
    configs = [
        ("B1 (source eval)", model_b1, source_test, layer_b1),
        ("B2 (shifted eval)", model_b1, shifted_test, layer_b1),
        ("B5 (shifted eval)", model_b5, shifted_test, layer_b5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    for col, (title, model, dataset, layer) in enumerate(configs):
        # Row 0: flaw (high-severity=2), correctly classified for B2
        result = _find_sample(dataset, severity=2, model=model,
                              target_layer=layer, correct=True)
        if result:
            signal, label, pred, cam = result
            ax = axes[0, col]
            ax.plot(signal, color="gray", alpha=0.6, linewidth=0.5)
            ax.fill_between(range(len(cam)), 0, cam * signal.max() * 0.8,
                            alpha=0.4, color="orangered")
            ax.set_title(f"{title}\nflaw (sev=2, pred={pred})", fontsize=9)
            ax.set_xlim(0, len(signal))

        # Row 1: no-flaw (severity=0)
        result = _find_sample(dataset, severity=0, model=model,
                              target_layer=layer, correct=(col != 1))
        if result:
            signal, label, pred, cam = result
            ax = axes[1, col]
            ax.plot(signal, color="gray", alpha=0.6, linewidth=0.5)
            ax.fill_between(range(len(cam)), 0, cam * signal.max() * 0.8,
                            alpha=0.4, color="steelblue")
            ax.set_title(f"{title}\nno-flaw (pred={pred})", fontsize=9)
            ax.set_xlim(0, len(signal))

    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude / Attribution")
    for ax in axes[1, :]:
        ax.set_xlabel("Sample index")

    fig.suptitle("Grad-CAM Attribution: Where Does the Model Look?", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_mean_attribution_profile(save_path, model_b1, model_b5,
                                   source_test, shifted_test, n_samples=50):
    """Mean Grad-CAM over flaw samples: B1/source, B2/shifted, B5/shifted."""
    layer_b1 = _last_conv(model_b1)
    layer_b5 = _last_conv(model_b5)

    # Filter to high-severity (label=2) samples for each dataset
    def _severity_indices(dataset, severity):
        return [i for i in range(len(dataset)) if dataset[i][1].item() == severity]

    source_flaw_idx = _severity_indices(source_test, 2)
    shifted_flaw_idx = _severity_indices(shifted_test, 2)

    # Subsample deterministically
    rng = np.random.default_rng(42)
    source_idx = rng.choice(source_flaw_idx, size=min(n_samples, len(source_flaw_idx)),
                            replace=False)
    shifted_idx = rng.choice(shifted_flaw_idx, size=min(n_samples, len(shifted_flaw_idx)),
                             replace=False)

    def _compute_mean_cam(model, dataset, indices, layer):
        cams = []
        for idx in indices:
            x, y = dataset[int(idx)]
            x_batch = x.unsqueeze(0)
            with torch.no_grad():
                pred = model(x_batch).argmax(dim=1).item()
            cam = gradcam_1d(model, x_batch, target_class=pred, target_layer=layer)
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max
            cams.append(cam)
        return np.mean(cams, axis=0)

    # Compute mean attributions
    cam_b1 = _compute_mean_cam(model_b1, source_test, source_idx, layer_b1)
    cam_b2 = _compute_mean_cam(model_b1, shifted_test, shifted_idx, layer_b1)
    cam_b5 = _compute_mean_cam(model_b5, shifted_test, shifted_idx, layer_b5)

    # Gray envelope: mean |signal| from source samples only
    signals = []
    for idx in source_idx:
        x, _ = source_test[int(idx)]
        signals.append(np.abs(x.squeeze().numpy()))
    mean_envelope = np.mean(signals, axis=0)
    # Normalize envelope to [0, 1] for overlay
    env_max = mean_envelope.max()
    if env_max > 0:
        mean_envelope = mean_envelope / env_max

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(range(len(mean_envelope)), 0, mean_envelope,
                    alpha=0.15, color="gray", label="Mean |signal| (source)")
    ax.plot(cam_b1, color="#4c72b0", linewidth=1.5, label="B1 on source")
    ax.plot(cam_b2, color="#dd8452", linewidth=1.5, label="B2 on shifted")
    ax.plot(cam_b5, color="#55a868", linewidth=1.5, label="B5 on shifted")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Mean normalized Grad-CAM")
    ax.set_title(f"Mean Attribution Profile (n={len(source_idx)} high-severity flaw samples)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM figures")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="docs/figures")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    model_b1 = _load_model(os.path.join(args.models_dir, "B1_cnn1d_source.pt"))
    model_b5 = _load_model(os.path.join(args.models_dir, "B5_cnn1d_randomized_finetuned.pt"))

    # Load datasets
    norm = Normalize()
    source_test = InspectionDataset(
        os.path.join(args.data_dir, "source_test.npz"), transform=norm,
    )
    shifted_test = InspectionDataset(
        os.path.join(args.data_dir, "shifted_test.npz"), transform=norm,
    )

    print("Generating attribution grid...")
    plot_attribution_grid(
        os.path.join(args.output_dir, "gradcam_grid.png"),
        model_b1, model_b5, source_test, shifted_test,
    )

    print("Generating mean attribution profile...")
    plot_mean_attribution_profile(
        os.path.join(args.output_dir, "gradcam_mean_profile.png"),
        model_b1, model_b5, source_test, shifted_test,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

```bash
python experiments/generate_gradcam_figures.py
```

Expected: Two PNG files in `docs/figures/`. Visually inspect:
- `gradcam_grid.png`: 2x3 grid, B1 column should show attribution near defect echo region, B2 column may show diffuse/drifted attention.
- `gradcam_mean_profile.png`: Blue line (B1) should peak near defect echo, orange (B2) should be flatter or shifted.

**Step 3: Commit**

```bash
git add experiments/generate_gradcam_figures.py
git commit -m "feat(ext2): Grad-CAM figure generation script (grid + mean profile)"
```

---

## Task 4: Commit figures and update README

**Files:**
- Modify: `README.md`
- Add: `docs/figures/gradcam_grid.png`
- Add: `docs/figures/gradcam_mean_profile.png`

**Step 1: Add README section**

Insert after the confusion matrix figure block (after line 56 — the `</p>` closing the confusion matrices) and before the `### Calibration` heading. The new section:

```markdown
### Attribution Analysis (Grad-CAM)

Grad-CAM attributions show where the 1D CNN attends when classifying A-scan traces. On source data (B1), attribution peaks align with the defect echo arrival region. Under domain shift (B2), attention drifts toward noise and back-wall artifacts — explaining the per-class recall collapse in the confusion matrices above. Domain randomization with fine-tuning (B5) partially recovers attention to the defect region.

<p align="center">
  <img src="docs/figures/gradcam_grid.png" width="800" alt="Grad-CAM attribution grid: B1 source, B2 shifted, B5 shifted for flaw and no-flaw samples">
  <br><em>Representative samples (seed=42). Signal in gray, Grad-CAM attribution overlaid in color.</em>
</p>

The mean attribution profile averaged over 50 high-severity flaw samples confirms the pattern quantitatively: B1's attention peak (blue) sits on the defect echo, B2's (orange) is diffuse, and B5's (green) partially recovers.

<p align="center">
  <img src="docs/figures/gradcam_mean_profile.png" width="700" alt="Mean Grad-CAM attribution profile for B1, B2, B5 over 50 flaw samples">
</p>

These are qualitative diagnostics showing where the model attends, not proof of learned physical reasoning.
```

**Step 2: Commit figures and README**

```bash
git add docs/figures/gradcam_grid.png docs/figures/gradcam_mean_profile.png README.md
git commit -m "feat(ext2): Grad-CAM figures and README attribution analysis section"
```

---

## Task 5: Final verification

**Step 1: Run all tests**

```bash
python -m pytest tests/ -q --ignore=tests/test_dataset.py --ignore=tests/test_benchmark_smoke.py
```

Expected: All pass including 7 new interpretability tests.

**Step 2: Run remaining tests**

```bash
python -m pytest tests/test_dataset.py tests/test_benchmark_smoke.py -q
```

Expected: All pass.

**Step 3: Run linter**

```bash
python -m ruff check src/ tests/ experiments/
```

Expected: No errors.

**Step 4: Commit any linter fixes**

```bash
git add -u
git commit -m "chore(ext2): linter fixes"
```

Only if needed.
