"""
baselines/frequency.py
───────────────────────
FEDformer (frequency-domain attention) and TimesNet baselines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FEDformer (simplified — random frequency mode selection)
# ---------------------------------------------------------------------------
class FEDformerBlock(nn.Module):
    def __init__(self, d_model: int, n_modes: int = 64) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.W_r = nn.Parameter(torch.randn(n_modes, d_model, d_model) * 0.02)
        self.W_i = nn.Parameter(torch.randn(n_modes, d_model, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)              # (B, T//2+1, d)
        modes = min(self.n_modes, xf.shape[1])
        out = xf.clone()
        out[:, :modes] = (
            torch.einsum("bmd,mdo->bmo", xf[:, :modes].real, self.W_r[:modes])
            + 1j * torch.einsum("bmd,mdo->bmo", xf[:, :modes].imag, self.W_i[:modes])
        )
        return torch.fft.irfft(out, n=T, dim=1)


class FEDformer(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 d_model: int = 128, n_modes: int = 64, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            FEDformerBlock(d_model, n_modes) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ff    = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_layers)
        ])
        self.proj  = nn.Linear(d_model * seq_len, pred_len)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.input_proj(x.unsqueeze(-1))
        for blk, norm, ff in zip(self.blocks, self.norms, self.ff):
            h = norm(h + self.drop(blk(h)))
            h = norm(h + self.drop(ff(h)))
        return self.proj(h.reshape(B, -1))


# ---------------------------------------------------------------------------
# TimesNet
# ---------------------------------------------------------------------------
class TimesBlock(nn.Module):
    """
    Reshape 1-D series into 2-D (period × period_len) and apply 2-D conv.
    """

    def __init__(self, d_model: int, d_ff: int, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        xf = torch.fft.rfft(x.mean(-1), dim=-1)
        amps = xf.abs()[:, 1:]                              # skip DC
        top_periods = amps.topk(min(self.top_k, amps.size(-1)), dim=-1).indices + 1

        outputs = []
        for p in range(top_periods.size(-1)):
            period = int(top_periods[:, p].float().mean().round().item())
            period = max(period, 1)
            period_len = math.ceil(T / period)
            pad_len = period * period_len - T
            xp = F.pad(x, (0, 0, 0, pad_len))             # (B, T+pad, d)
            xp = xp.reshape(B, period_len, period, D).permute(0, 3, 1, 2)  # (B,d,pl,p)
            xp = self.conv(xp)
            xp = xp.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :T]
            outputs.append(xp)

        return torch.stack(outputs, dim=-1).mean(-1)


class TimesNet(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 d_model: int = 64, d_ff: int = 128, n_layers: int = 2,
                 top_k: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks  = nn.ModuleList([TimesBlock(d_model, d_ff, top_k) for _ in range(n_layers)])
        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.proj    = nn.Linear(d_model * seq_len, pred_len)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.input_proj(x.unsqueeze(-1))
        for blk, norm in zip(self.blocks, self.norms):
            h = norm(h + self.drop(blk(h)))
        return self.proj(h.reshape(B, -1))
