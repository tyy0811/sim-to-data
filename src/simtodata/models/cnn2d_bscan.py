"""Small 2D CNN for binary B-scan flaw detection.

Operates on (1, 64, 64) resized B-scan images. ~30K parameters.
Designed for CPU training on synthetic B-scan datasets.
"""

import torch.nn as nn


class BscanCNN(nn.Module):
    """2D CNN for binary B-scan classification (flaw / no-flaw)."""

    def __init__(self, channels=(16, 32, 64), fc_hidden=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels[2], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
