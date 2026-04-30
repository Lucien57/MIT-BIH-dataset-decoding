from typing import Dict

import torch.nn as nn

from models.NM2019 import NM2019
from models.Transformer import ECGTransformer
from models.LSTM import ECGLSTM
from models.LightCNN import LightCNN
from models.TransformerCNN import LightTransformerCNN


def load_model(model_cfg: Dict) -> nn.Module:
    name = model_cfg.get("name", "NM2019")
    cfg = dict(model_cfg)
    cfg.pop("name", None)

    if name == "NM2019":
        return NM2019(**cfg)
    if name == "ECGTransformer":
        return ECGTransformer(**cfg)
    if name == "ECGLSTM":
        return ECGLSTM(**cfg)
    if name == "LightCNN":
        return LightCNN(**cfg)
    if name == "LightTransformerCNN":
        return LightTransformerCNN(**cfg)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
