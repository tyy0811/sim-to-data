"""Integration tests: config -> generate -> train -> evaluate."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simtodata.data.generate import generate_dataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.metrics import compute_all_metrics
from simtodata.features.extract import extract_features_batch
from simtodata.models.baselines import create_baseline
from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.predict import predict_batch
from simtodata.models.train import train_model


def test_baseline_pipeline(source_regime):
    """Generate -> extract features -> fit baseline -> predict -> metrics."""
    regime = source_regime
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    data = generate_dataset(regime, 300, seed=42, class_distribution=class_dist)

    features = extract_features_batch(data["signals"])
    assert features.shape == (300, 11)

    clf = create_baseline("gradient_boosting")
    clf.fit(features[:200], data["labels"][:200])
    preds = clf.predict(features[200:])
    probs = clf.predict_proba(features[200:])
    metrics = compute_all_metrics(data["labels"][200:], preds, probs)

    assert "macro_f1" in metrics
    assert 0 <= metrics["macro_f1"] <= 1
    assert np.isfinite(metrics["ece"])


def test_cnn_pipeline(source_regime):
    """Generate -> dataset -> train CNN -> predict -> metrics."""
    regime = source_regime
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    data = generate_dataset(regime, 120, seed=42, class_distribution=class_dist)

    normalize = Normalize()
    signals = torch.from_numpy(data["signals"]).unsqueeze(1)
    labels = torch.from_numpy(data["labels"])
    for i in range(len(signals)):
        signals[i] = normalize(signals[i])

    train_ds = TensorDataset(signals[:80], labels[:80])
    test_ds = TensorDataset(signals[80:], labels[80:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = DefectCNN1D()
    model, history = train_model(model, train_loader, epochs=3, lr=1e-3)

    preds, probs, true_labels = predict_batch(model, test_loader)
    metrics = compute_all_metrics(true_labels, preds, probs)

    assert "macro_f1" in metrics
    assert "auroc" in metrics
    assert len(history["train_loss"]) == 3
