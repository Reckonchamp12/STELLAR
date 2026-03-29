"""
stellar/model.py
────────────────
Full STELLAR model — assembles all five components into a unified
probabilistic forecasting architecture.

Forward signature
-----------------
    x    : (B, seq_len)  — raw (pre-normalisation) input
    returns
    mean : (B, pred_len) — point forecast in original scale
    var  : (B, pred_len) — predictive variance in original scale
    kl   : scalar        — KL divergence for ELBO training
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import RevIN, AdaptiveSpectralFilter, KoopmanDynamics, MRTP, NPHead


class STELLAR(nn.Module):
    """
    STELLAR — Spectral-Temporal Ensemble Learning with Latent Adaptive
    Representations.

    Five inductive-bias paths are combined via a learned gate:
      1. Koopman latent dynamics (long-range linear propagation)
      2. Adaptive spectral filter → linear residual
      3. Multi-resolution temporal patches
      4. Neural-Process probabilistic head (calibrated uncertainty)
      5. Gated prediction mixture (sample-adaptive routing)

    Parameters
    ----------
    seq_len  : input sequence length
    pred_len : forecast horizon
    d        : hidden dimension for ASF projection and MRTP
    d_koop   : latent dimension for Koopman encoder
    n_modes  : number of Koopman spectral modes
    d_lat    : latent variable dimension for NPHead
    """

    def __init__(
        self,
        seq_len:  int = 96,
        pred_len: int = 24,
        d:        int = 64,
        d_koop:   int = 64,
        n_modes:  int = 32,
        d_lat:    int = 32,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len

        self.revin    = RevIN()
        self.asf      = AdaptiveSpectralFilter(seq_len, d_hid=64)
        self.asf_proj = nn.Linear(seq_len, d)
        self.koopman  = KoopmanDynamics(seq_len, pred_len, d=d_koop, n_modes=n_modes)
        self.mrtp     = MRTP(seq_len, d=d)

        d_ctx = d * 2   # asf_proj(d) + mrtp(d)

        self.linear = nn.Linear(seq_len, pred_len, bias=False)
        self.nphead = NPHead(d_ctx, pred_len, d_lat=d_lat, d_hid=128)
        self.gate   = nn.Sequential(nn.Linear(d_ctx, 3), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor):
        xn = self.revin.norm(x)                     # instance-normed input

        # ── Path 1 : Adaptive Spectral ──
        x_sp = self.asf(xn)                         # (B, T)
        f_sp = self.asf_proj(x_sp)                  # (B, d)

        # ── Path 2 : Koopman dynamics ──
        koop = self.koopman(xn)                     # (B, pred_len)

        # ── Path 3 : Multi-resolution patches ──
        f_mr = self.mrtp(xn)                        # (B, d)

        # ── Path 4 : Linear residual ──
        lin = self.linear(xn)                       # (B, pred_len)

        # ── Fuse context ──
        ctx = torch.cat([f_sp, f_mr], dim=-1)       # (B, 2d)

        # ── Path 5 : Gate ──
        g            = self.gate(ctx)               # (B, 3)
        g_k, g_l, g_np = g[:, 0:1], g[:, 1:2], g[:, 2:3]

        # ── Probabilistic head ──
        np_mu, np_var, kl = self.nphead(ctx)        # (B,P), (B,P), scalar

        # ── Combine & denormalise ──
        mean_n = g_k * koop + g_l * lin + g_np * np_mu
        mean   = self.revin.denorm(mean_n)

        sig2 = self.revin._sig ** 2                 # (B, 1)
        var  = np_var * sig2                        # (B, pred_len)

        return mean, var, kl
