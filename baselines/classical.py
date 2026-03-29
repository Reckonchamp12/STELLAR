"""
baselines/classical.py
──────────────────────
NLinear and DLinear baselines (Zeng et al., 2023).
"""

import torch
import torch.nn as nn


class NLinear(nn.Module):
    """Non-stationary Linear: subtracts the last value before projecting."""

    def __init__(self, seq_len: int = 96, pred_len: int = 24) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:]
        return self.linear(x - last) + last


class DLinear(nn.Module):
    """Decomposition Linear: trend + seasonal linear projections."""

    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 kernel_size: int = 25) -> None:
        super().__init__()
        self.trend_proj    = nn.Linear(seq_len, pred_len)
        self.seasonal_proj = nn.Linear(seq_len, pred_len)
        self.kernel_size   = kernel_size
        self.avg_pool      = nn.AvgPool1d(kernel_size=kernel_size,
                                          stride=1,
                                          padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        trend    = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        trend    = trend[:, :x.size(1)]           # trim padding artefacts
        seasonal = x - trend
        return self.trend_proj(trend) + self.seasonal_proj(seasonal)
