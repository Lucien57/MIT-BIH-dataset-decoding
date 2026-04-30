"""Light parallel Transformer-CNN model for MIT-BIH heartbeat-window classification."""

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
        return self.dropout(x + self.pe[:, :x.size(1), :])


class CNNBranch(nn.Module):
    def __init__(self, in_channels: int, channels=(24, 48, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out, k, s in zip(channels, (9, 7, 5), (1, 2, 2)):
            layers.extend([
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.out_dim = 2 * c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.cat([self.pool_avg(x).squeeze(-1), self.pool_max(x).squeeze(-1)], dim=1)


class TransformerBranch(nn.Module):
    def __init__(self, in_channels: int, input_length: int, d_model: int = 48, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 96, patch_size: int = 6,
                 patch_stride: int = 3, dropout: float = 0.1):
        super().__init__()
        self.patch = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                               stride=patch_stride, padding=patch_size // 2, bias=False)
        token_len = math.floor((input_length + 2 * (patch_size // 2) - patch_size) / patch_stride + 1)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max(512, token_len + 8), dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).transpose(1, 2)
        x = self.pos(x)
        x = self.encoder(x)
        return self.norm(x).mean(dim=1)


class LightTransformerCNN(nn.Module):
    """Parallel CNN + Transformer model.

    CNN branch: local morphology and robust short-range patterns.
    Transformer branch: longer-range temporal interaction across the heartbeat window.
    The two features are concatenated and classified by a small MLP.
    """

    def __init__(self,
                 n_class: int = 5,
                 in_channels: int = 1,
                 input_length: int = 300,
                 cnn_channels=(24, 48, 64),
                 d_model: int = 48,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 96,
                 patch_size: int = 6,
                 patch_stride: int = 3,
                 dropout: float = 0.1,
                 head_dim: int = 128):
        super().__init__()
        self.cnn = CNNBranch(in_channels, channels=cnn_channels, dropout=dropout)
        self.trans = TransformerBranch(
            in_channels=in_channels,
            input_length=input_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            patch_size=patch_size,
            patch_stride=patch_stride,
            dropout=dropout,
        )
        fusion_dim = self.cnn.out_dim + self.trans.out_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_class),
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
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return torch.cat([self.cnn(x), self.trans(x)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_model(**kwargs) -> LightTransformerCNN:
    return LightTransformerCNN(**kwargs)
