"""
PyTorch adaptation of the residual 1-D CNN used by Hannun et al. (Nature Medicine, 2019).

The original network makes dense sequence-level predictions on long single-lead ECG records.
For the MIT-BIH heartbeat-window task here, the residual stack is retained but the output head is
changed to global temporal pooling + one linear classifier, producing one label per 300-point beat.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    """Conv1d with TensorFlow/Keras-like 'same' padding, including stride > 1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_len = x.size(-1)
        out_len = math.ceil(in_len / self.stride)
        pad_needed = max((out_len - 1) * self.stride + self.kernel_size - in_len, 0)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ResidualBlock1D(nn.Module):
    """Pre-activation residual block with two convolutional layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dropout: float, block_index: int,
                 shortcut: str = "maxpool_zero_pad"):
        super().__init__()
        self.stride = int(stride)
        self.block_index = int(block_index)
        self.shortcut = shortcut
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        self.conv1 = SamePadConv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size, stride=1)

        if shortcut == "projection":
            self.proj = SamePadConv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.proj = None

    def _shortcut(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        if self.proj is not None:
            shortcut = self.proj(x)
        else:
            shortcut = x
            if self.stride > 1:
                shortcut = F.max_pool1d(shortcut, kernel_size=self.stride,
                                        stride=self.stride, ceil_mode=True)
            if shortcut.size(-1) > out_len:
                shortcut = shortcut[..., :out_len]
            elif shortcut.size(-1) < out_len:
                shortcut = F.pad(shortcut, (0, out_len - shortcut.size(-1)))

            if self.out_channels > self.in_channels:
                pad_ch = self.out_channels - self.in_channels
                shortcut = F.pad(shortcut, (0, 0, 0, pad_ch))
            elif self.out_channels < self.in_channels:
                shortcut = shortcut[:, :self.out_channels, :]
        return shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity_input = x

        # The first conv in the first residual block is special-cased in the original code.
        if self.block_index == 0:
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu(self.bn1(x)))

        out = self.conv2(self.dropout(self.relu(self.bn2(out))))
        shortcut = self._shortcut(identity_input, out.size(-1))
        return out + shortcut


class NM2019(nn.Module):
    """Nature-Medicine-2019-style residual CNN for five-class heartbeat classification."""

    def __init__(self,
                 n_class: int = 5,
                 in_channels: int = 1,
                 conv_filter_length: int = 16,
                 conv_num_filters_start: int = 32,
                 conv_subsample_lengths: Optional[List[int]] = None,
                 conv_num_skip: int = 2,
                 conv_increase_channels_at: int = 4,
                 conv_dropout: float = 0.2,
                 shortcut: str = "maxpool_zero_pad"):
        super().__init__()
        if conv_num_skip != 2:
            raise ValueError("This implementation follows the NM2019 two-conv residual block design; set conv_num_skip=2.")
        if conv_subsample_lengths is None:
            conv_subsample_lengths = [1, 2] * 8

        self.n_class = int(n_class)
        self.in_channels = int(in_channels)
        self.conv_subsample_lengths = list(conv_subsample_lengths)
        self.conv_increase_channels_at = int(conv_increase_channels_at)
        self.conv_num_filters_start = int(conv_num_filters_start)

        self.stem = SamePadConv1d(
            in_channels=in_channels,
            out_channels=conv_num_filters_start,
            kernel_size=conv_filter_length,
            stride=1,
        )
        self.stem_bn = nn.BatchNorm1d(conv_num_filters_start)
        self.relu = nn.ReLU(inplace=True)

        blocks = []
        in_ch = conv_num_filters_start
        for index, stride in enumerate(self.conv_subsample_lengths):
            out_ch = self._filters_at_block(index)
            blocks.append(ResidualBlock1D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=conv_filter_length,
                stride=stride,
                dropout=conv_dropout,
                block_index=index,
                shortcut=shortcut,
            ))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.final_bn = nn.BatchNorm1d(in_ch)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, n_class)
        self._init_weights()

    def _filters_at_block(self, index: int) -> int:
        scale = 2 ** int(index / self.conv_increase_channels_at)
        return scale * self.conv_num_filters_start

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.stem_bn(self.stem(x)))
        x = self.blocks(x)
        x = self.relu(self.final_bn(x))
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(features)


def build_model(**kwargs) -> NM2019:
    """Factory used by util/load_model.py."""
    return NM2019(**kwargs)
