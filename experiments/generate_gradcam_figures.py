"""Generate Grad-CAM attribution figures for domain shift analysis.

Figure A: 2x3 attribution grid (B1 source, B2 shifted, B5 shifted) x (flaw, no-flaw)
Figure B: Mean attribution profile (B1/B2/B5 averaged over 50 flaw samples)
"""

import argparse
import logging
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.interpretability import gradcam_1d
from simtodata.models.cnn1d import DefectCNN1D

logger = logging.getLogger(__name__)


def _load_model(path):
    """Load a DefectCNN1D from a state dict checkpoint.

    The saved checkpoints use a 3-block architecture that predates the
    current 4-block default in model_cnn1d.yaml.
    """
    model = DefectCNN1D(
        channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64, pool_size=1,
    )
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
    logger.warning(
        "No %s sample found for severity=%d; falling back to any "
        "correctness. Panel may not show intended behavior.",
        "correctly classified" if correct else "misclassified",
        severity,
    )
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
