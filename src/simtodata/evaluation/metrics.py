"""Evaluation metrics: F1, AUROC, ECE, per-class precision/recall."""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score


NUM_CLASSES = 3
CLASS_LABELS = list(range(NUM_CLASSES))


def compute_macro_f1(y_true, y_pred):
    """Compute macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", labels=CLASS_LABELS, zero_division=0))


def compute_auroc(y_true, y_proba):
    """Compute macro-averaged AUROC (one-vs-rest).

    Returns nan if fewer than 2 classes are present in y_true
    (AUROC is undefined in that case).
    """
    try:
        return float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=CLASS_LABELS
        ))
    except ValueError:
        return float("nan")


def compute_ece(y_true, y_proba, n_bins=10):
    """Compute Expected Calibration Error."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    return float(ece / len(y_true))


def compute_per_class_metrics(y_true, y_pred):
    """Compute per-class precision, recall, and F1.

    Always returns arrays of length NUM_CLASSES, even if some classes
    are absent from the eval split.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=CLASS_LABELS, zero_division=0
    )
    return {
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "f1": [float(f) for f in f1],
    }


def compute_all_metrics(y_true, y_pred, y_proba):
    """Compute all metrics in a single dict."""
    return {
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "auroc": compute_auroc(y_true, y_proba),
        "ece": compute_ece(y_true, y_proba),
        "per_class": compute_per_class_metrics(y_true, y_pred),
    }
