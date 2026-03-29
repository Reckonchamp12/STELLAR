"""
stellar/losses.py
─────────────────
Loss functions and evaluation metrics for probabilistic forecasting.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def gaussian_nll(mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood."""
    return (0.5 * torch.log(var) + 0.5 * (y - mean).pow(2) / var).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def crps_gaussian(mu: torch.Tensor, sig: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form CRPS for N(μ, σ²)."""
    normal = torch.distributions.Normal(0.0, 1.0)
    z = (y - mu) / sig.clamp(min=1e-8)
    crps = sig * (
        z * (2.0 * normal.cdf(z) - 1.0)
        + 2.0 * normal.log_prob(z).exp()
        - 1.0 / math.sqrt(math.pi)
    )
    return crps.mean().item()


def compute_metrics(
    pred: torch.Tensor,
    var: torch.Tensor,
    tgt: torch.Tensor,
    mad: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Return (MAE, RMSE, sMAPE%, MASE, CRPS, NLL).

    Parameters
    ----------
    pred : point forecast  (B, P)
    var  : predictive variance (B, P)  — must be positive
    tgt  : ground-truth          (B, P)
    mad  : mean absolute diff on training set (MASE denominator)
    """
    mae   = torch.abs(pred - tgt).mean().item()
    rmse  = torch.sqrt(((pred - tgt) ** 2).mean()).item()
    smape = (
        2.0 * torch.abs(pred - tgt) / (torch.abs(pred) + torch.abs(tgt) + 1e-8)
    ).mean().item() * 100.0
    mase  = mae / (mad + 1e-8)
    std   = var.clamp(min=1e-10).sqrt()
    crps  = crps_gaussian(pred, std, tgt)
    nll   = gaussian_nll(pred, var.clamp(min=1e-10), tgt).item()
    return mae, rmse, smape, mase, crps, nll
