"""2D CNN classifier on STFT spectrograms."""

import torch.nn as nn


class DefectCNN2D(nn.Module):
    """2D convolutional network for spectrogram classification."""

    def __init__(self, channels=(16, 32, 64), kernels=(3, 3, 3), fc_hidden=32,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], kernels[0]),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernels[1]),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernels[2]),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
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
