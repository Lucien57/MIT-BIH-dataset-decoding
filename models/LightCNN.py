"""Lightweight 1-D CNN baseline for MIT-BIH heartbeat-window classification."""

import torch
import torch.nn as nn


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return out + x


class LightCNN(nn.Module):
    """Compact depthwise-separable CNN.

    It is intentionally much lighter than NM2019, but still uses multi-stage temporal downsampling
    and residual shortcuts. This is a useful speed/accuracy baseline for the homework dataset.
    """

    def __init__(self,
                 n_class: int = 5,
                 in_channels: int = 1,
                 input_length: int = 300,
                 channels=(32, 64, 96, 128),
                 kernels=(9, 7, 5, 3),
                 strides=(1, 2, 2, 2),
                 dropout: float = 0.15,
                 head_dropout: float = 0.25):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out, k, s in zip(channels, kernels, strides):
            layers.append(SeparableConvBlock(c_in, int(c_out), int(k), int(s), dropout=dropout))
            c_in = int(c_out)
        self.encoder = nn.Sequential(*layers)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(2 * c_in, n_class),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        avg = self.pool_avg(x).squeeze(-1)
        mx = self.pool_max(x).squeeze(-1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_model(**kwargs) -> LightCNN:
    return LightCNN(**kwargs)
