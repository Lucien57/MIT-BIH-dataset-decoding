"""LSTM baseline for 300-point MIT-BIH heartbeat windows."""

import torch
import torch.nn as nn


class ECGLSTM(nn.Module):
    """Bidirectional LSTM classifier.

    For single-lead ECG, each time step has one feature. A small linear input projection is used
    before the LSTM so the recurrent layer sees a richer per-time-step representation.
    """

    def __init__(self,
                 n_class: int = 5,
                 in_channels: int = 1,
                 input_length: int = 300,
                 input_dim: int = 32,
                 hidden_size: int = 96,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2,
                 pooling: str = "last_mean"):
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_class),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)          # (B, T, C)
        x = self.input_proj(x)        # (B, T, input_dim)
        out, _ = self.lstm(x)         # (B, T, H * directions)
        if self.pooling == "mean":
            return out.mean(dim=1)
        if self.pooling == "max":
            return out.max(dim=1).values
        if self.pooling == "last":
            return out[:, -1, :]
        # last_mean is usually safer for short centered heartbeat windows.
        return 0.5 * (out[:, -1, :] + out.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(**kwargs) -> ECGLSTM:
    return ECGLSTM(**kwargs)
