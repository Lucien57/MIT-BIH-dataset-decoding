from typing import Dict, Optional, Sequence

import torch


class ECGAugment:
    """Lightweight ECG augmentations for heartbeat-window classification.

    Input and output are both tensors of shape (C, T). All transforms are intentionally modest;
    MIT-BIH windows are short heartbeat-centered segments, so aggressive temporal warping is risky.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.enable = bool(cfg.get("enable", False))
        self.noise_std = float(cfg.get("noise_std", 0.0) or 0.0)
        self.scale_range = cfg.get("scale_range", [1.0, 1.0])
        self.shift_max = int(cfg.get("shift_max", 0) or 0)
        self.time_mask_ratio = float(cfg.get("time_mask_ratio", 0.0) or 0.0)
        self.p = float(cfg.get("p", 1.0))

    def _rand_uniform(self, low: float, high: float, device: torch.device) -> torch.Tensor:
        return torch.empty(1, device=device).uniform_(float(low), float(high))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable or torch.rand(1, device=x.device).item() > self.p:
            return x

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if isinstance(self.scale_range, Sequence) and len(self.scale_range) == 2:
            low, high = float(self.scale_range[0]), float(self.scale_range[1])
            if low != 1.0 or high != 1.0:
                x = x * self._rand_uniform(low, high, x.device)

        if self.shift_max > 0:
            shift = int(torch.randint(-self.shift_max, self.shift_max + 1, (1,), device=x.device).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        if self.time_mask_ratio > 0:
            time_len = x.size(-1)
            mask_len = int(round(time_len * self.time_mask_ratio))
            if mask_len > 0 and mask_len < time_len:
                start = int(torch.randint(0, time_len - mask_len + 1, (1,), device=x.device).item())
                x = x.clone()
                x[..., start:start + mask_len] = 0.0

        return x
