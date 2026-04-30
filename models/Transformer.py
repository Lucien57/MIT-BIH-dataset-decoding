"""Lightweight Transformer encoder for 300-point MIT-BIH heartbeat windows."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ECGTransformer(nn.Module):
    """Small Transformer for heartbeat-window classification.

    The input is first converted to short local tokens by a strided Conv1d patch embedding.
    This is more stable than feeding all 300 raw points as individual tokens.
    """

    def __init__(self,
                 n_class: int = 5,
                 in_channels: int = 1,
                 input_length: int = 300,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 128,
                 patch_size: int = 6,
                 patch_stride: int = 3,
                 dropout: float = 0.1,
                 pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.patch = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                               stride=patch_stride, padding=patch_size // 2, bias=False)
        token_len = math.floor((input_length + 2 * (patch_size // 2) - patch_size) / patch_stride + 1)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max(512, token_len + 8), dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_class),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.patch(x).transpose(1, 2)  # (B, L, D)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        if self.pooling == "max":
            return x.max(dim=1).values
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_model(**kwargs) -> ECGTransformer:
    return ECGTransformer(**kwargs)
