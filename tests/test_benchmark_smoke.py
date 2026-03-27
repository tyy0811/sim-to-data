"""End-to-end smoke test: generate -> train -> predict -> metrics."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from simtodata.data.generate import generate_dataset
from simtodata.data.transforms import Normalize
from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.train import train_model
from simtodata.models.predict import predict_batch
from simtodata.simulator.regime import RegimeConfig


def _source_regime():
    return RegimeConfig(
        name="source",
        thickness_mm=(10.0, 30.0),
        velocity_ms=(5800.0, 6200.0),
        attenuation_np_mm=(0.01, 0.05),
        center_freq_mhz=(2.0, 5.0),
        pulse_sigma_us=(0.5, 1.5),
        defect_depth_mm=(2.0, 28.0),
        defect_reflectivity=(0.1, 0.8),
        snr_db=(20.0, 40.0),
        baseline_drift=(0.0, 0.0),
        gain_variation=(1.0, 1.0),
        jitter_samples=(0, 0),
        dropout_n_gaps=(0, 0),
        dropout_gap_length=(0, 0),
    )


def test_end_to_end_smoke():
    """Full pipeline smoke test with tiny data. Must complete in <60s."""
    regime = _source_regime()
    class_dist = {"no_defect": 0.33, "low_severity": 0.33, "high_severity": 0.34}
    data = generate_dataset(regime, 90, seed=42, class_distribution=class_dist)

    normalize = Normalize()
    signals = torch.from_numpy(data["signals"]).unsqueeze(1)
    labels = torch.from_numpy(data["labels"])
    for i in range(len(signals)):
        signals[i] = normalize(signals[i])

    train_ds = TensorDataset(signals[:60], labels[:60])
    val_ds = TensorDataset(signals[60:80], labels[60:80])
    test_ds = TensorDataset(signals[80:], labels[80:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = DefectCNN1D()
    model, history = train_model(model, train_loader, val_loader, epochs=2, lr=1e-3)

    preds, probs, true_labels = predict_batch(model, test_loader)

    assert preds.shape == (10,)
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert all(p in {0, 1, 2} for p in preds)
