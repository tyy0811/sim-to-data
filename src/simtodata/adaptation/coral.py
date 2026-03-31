"""CORAL: Correlation Alignment for domain adaptation.

Aligns second-order feature statistics (covariance) between source
and target domains. Simplest principled DA method — ~30 lines of core logic.

Sun & Saenko (2016). Deep CORAL: Correlation Alignment for Deep
Domain Adaptation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def coral_loss(
    source_features: torch.Tensor, target_features: torch.Tensor,
) -> torch.Tensor:
    """Compute CORAL loss between source and target feature batches.

    CORAL loss = (1 / 4d^2) * ||C_s - C_t||^2_F

    Args:
        source_features: (N_s, D) features from source domain.
        target_features: (N_t, D) features from target domain.

    Returns:
        Scalar CORAL loss.
    """
    d = source_features.shape[1]

    src_centered = source_features - source_features.mean(dim=0, keepdim=True)
    tgt_centered = target_features - target_features.mean(dim=0, keepdim=True)

    n_s = source_features.shape[0]
    n_t = target_features.shape[0]
    cov_s = (src_centered.T @ src_centered) / max(n_s - 1, 1)
    cov_t = (tgt_centered.T @ tgt_centered) / max(n_t - 1, 1)

    diff = cov_s - cov_t
    return (diff * diff).sum() / (4 * d * d)


class FeatureExtractor:
    """Hook-based feature extractor for a named layer.

    Attaches a forward hook and stores activations. No model modification.

    Usage:
        extractor = FeatureExtractor(model, 'features')
        _ = model(x)
        feats = extractor.get()  # (B, D) after flatten
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.features: Optional[torch.Tensor] = None
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features = output

    def get(self) -> Optional[torch.Tensor]:
        return self.features


def train_with_coral(
    model: nn.Module,
    source_loader,
    target_loader,
    optimizer: torch.optim.Optimizer,
    feature_layer: str,
    coral_weight: float = 1.0,
    n_epochs: int = 20,
    device: str = "cpu",
) -> list:
    """Fine-tune model with CORAL regularization.

    Total loss = CE_loss + coral_weight * coral_loss.
    Target loader is unlabeled (features only for covariance alignment).

    Args:
        model: pre-trained CNN.
        source_loader: labeled source domain DataLoader.
        target_loader: unlabeled target domain DataLoader.
        optimizer: optimizer for fine-tuning.
        feature_layer: name of layer to hook (e.g. 'features').
        coral_weight: lambda for CORAL loss.
        n_epochs: fine-tuning epochs.
        device: 'cuda' or 'cpu'.

    Returns:
        List of per-epoch loss dicts.
    """
    extractor = FeatureExtractor(model, feature_layer)
    ce_loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_ce, epoch_coral, n_batches = 0.0, 0.0, 0

        target_iter = iter(target_loader)
        for src_x, src_y in source_loader:
            try:
                tgt_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_x, _ = next(target_iter)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            # Source forward
            src_out = model(src_x)
            src_feats = extractor.get()
            ce = ce_loss_fn(src_out, src_y)

            # Target forward
            _ = model(tgt_x)
            tgt_feats = extractor.get()

            # Flatten features if multi-dimensional (e.g. conv output)
            if src_feats.dim() > 2:
                src_feats = src_feats.flatten(1)
            if tgt_feats.dim() > 2:
                tgt_feats = tgt_feats.flatten(1)

            c_loss = coral_loss(src_feats, tgt_feats)
            total = ce + coral_weight * c_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_ce += ce.item()
            epoch_coral += c_loss.item()
            n_batches += 1

        history.append({
            "epoch": epoch,
            "ce_loss": epoch_ce / max(n_batches, 1),
            "coral_loss": epoch_coral / max(n_batches, 1),
        })

    return history
