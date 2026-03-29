"""
baselines/recurrent.py
───────────────────────
LSTM and DeepAR baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.proj(out[:, -1, :])


class DeepAR(nn.Module):
    """Simplified DeepAR — LSTM encoder with Gaussian output head."""

    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm   = nn.LSTM(1, hidden, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.mu_h   = nn.Linear(hidden, pred_len)
        self.sig_h  = nn.Linear(hidden, pred_len)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x.unsqueeze(-1))
        h      = out[:, -1, :]
        mu     = self.mu_h(h)
        sig    = F.softplus(self.sig_h(h)) + 1e-6
        return mu, sig
