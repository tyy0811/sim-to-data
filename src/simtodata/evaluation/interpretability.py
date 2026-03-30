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
