"""1D CNN classifier for defect severity detection."""

import torch.nn as nn


class DefectCNN1D(nn.Module):
    """1D convolutional network for A-scan classification."""

    def __init__(self, channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels[0], kernels[0], stride=2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[0], channels[1], kernels[1], stride=1),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels[1], channels[2], kernels[2], stride=1),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels[2], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
