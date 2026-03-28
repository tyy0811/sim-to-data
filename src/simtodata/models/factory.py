"""Model factory: config -> nn.Module."""

import yaml

from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.cnn2d_spectrogram import DefectCNN2D


def model_from_config(config_path: str):
    """Instantiate a model from a YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    arch = config["architecture"]
    if arch["type"] == "cnn_1d":
        return DefectCNN1D(
            channels=tuple(arch["channels"]),
            kernels=tuple(arch["kernels"]),
            fc_hidden=arch["fc_hidden"],
            dropout=arch["dropout"],
            num_classes=arch["num_classes"],
            pool_size=arch.get("pool_size", 4),
        )
    if arch["type"] == "cnn_2d_spectrogram":
        return DefectCNN2D(
            channels=tuple(arch["channels"]),
            kernels=tuple(arch["kernels"]),
            fc_hidden=arch["fc_hidden"],
            dropout=arch["dropout"],
            num_classes=arch["num_classes"],
        )
    raise ValueError(f"Unknown architecture type: {arch['type']}")
