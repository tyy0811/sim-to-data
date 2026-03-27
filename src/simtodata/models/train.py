"""Model training with early stopping and LR scheduling."""

import numpy as np
import torch
import torch.nn as nn
from simtodata.evaluation.metrics import compute_macro_f1


def _evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    n = 0
    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            n += len(labels)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / n
    f1 = compute_macro_f1(np.array(all_labels), np.array(all_preds))
    return avg_loss, f1


def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3,
                weight_decay=1e-4, patience=10, scheduler_patience=5,
                scheduler_factor=0.5, device="cpu"):
    """Train a model with optional early stopping and LR scheduling.

    Args:
        model: nn.Module to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (None to skip validation/early stopping).
        epochs: Maximum training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience (epochs without val improvement).
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR reduction factor.
        device: Device string.

    Returns:
        (model, history) where history is a dict of per-epoch metrics.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=scheduler_patience, factor=scheduler_factor
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)

        train_loss = total_loss / n
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, val_f1 = _evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_f1: {val_f1:.4f}"
                )
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
