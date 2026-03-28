"""1D CNN classifier for defect severity detection."""

import torch.nn as nn


class DefectCNN1D(nn.Module):
    """1D convolutional network for A-scan classification."""

    def __init__(self, channels=(32, 64, 128, 128), kernels=(7, 5, 3, 3), fc_hidden=128,
                 dropout=0.3, num_classes=3, pool_size=4):
        super().__init__()
        layers = []
        in_ch = 1
        for i, (out_ch, k) in enumerate(zip(channels, kernels)):
            stride = 2 if i == 0 else 1
            layers.extend([
                nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(pool_size))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * pool_size, fc_hidden),
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
