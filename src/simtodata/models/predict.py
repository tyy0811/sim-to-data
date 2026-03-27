"""Model inference and probability output."""

import numpy as np
import torch


def predict_batch(model, dataloader, device="cpu"):
    """Run inference on a DataLoader.

    Returns:
        (predictions, probabilities, labels) as numpy arrays.
    """
    model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs), np.concatenate(all_labels)
