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
