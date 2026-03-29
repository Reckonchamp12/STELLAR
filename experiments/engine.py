"""
experiments/engine.py
──────────────────────
Shared training / evaluation loop used by all experiment scripts.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data.dataset import load_dataset, make_loader
from stellar.losses import gaussian_nll, compute_metrics


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------
def train_and_eval(
    name: str,
    gpu_id: int,
    model_factory: Callable[[], nn.Module],
    *,
    seq_len:   int   = 96,
    pred_len:  int   = 24,
    batch:     int   = 64,
    lr:        float = 1e-3,
    epochs:    int   = 20,
    patience:  int   = 5,
    kl_weight: float = 1e-3,
    probabilistic: bool = True,   # True → model returns (mean, var, kl)
) -> dict:
    """
    Train `model_factory()` on dataset `name` and return a result dict.

    Parameters
    ----------
    probabilistic : if True the model is expected to return (mean, var, kl).
                    If False it returns a single tensor (point forecast) and
                    MAE is used as the training loss; CRPS/NLL are reported
                    as NaN.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] → GPU {gpu_id}")
    t0 = time.time()

    train_ds, val_ds, test_ds, sc, mad = load_dataset(name, seq_len, pred_len)
    tr_dl = make_loader(train_ds, batch, shuffle=True)
    va_dl = make_loader(val_ds,   batch, shuffle=False)
    te_dl = make_loader(test_ds,  batch, shuffle=False)

    model  = model_factory().to(device)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler()

    best_vloss  = float("inf")
    best_state  = None
    pat_counter = 0

    for ep in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────
        model.train()
        tloss = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                if probabilistic:
                    mu, var, kl = model(x)
                    loss = gaussian_nll(mu, var, y) + kl_weight * kl
                else:
                    mu   = model(x)
                    loss = torch.abs(mu - y).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tloss += loss.item()
        sched.step()

        # ── validate ───────────────────────────────────────────
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                with autocast():
                    if probabilistic:
                        mu, var, kl = model(x)
                        vloss += gaussian_nll(mu, var, y).item()
                    else:
                        mu    = model(x)
                        vloss += torch.abs(mu - y).mean().item()
        vloss /= max(len(va_dl), 1)

        if vloss < best_vloss - 1e-6:
            best_vloss  = vloss
            pat_counter = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_counter += 1
            if pat_counter >= patience:
                print(f"  [{name}] early stop @ ep {ep}")
                break

    # ── restore best & evaluate ────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    all_mu, all_var, all_y = [], [], []
    dummy_var = None
    with torch.no_grad():
        for x, y in te_dl:
            x = x.to(device)
            if probabilistic:
                mu, var, _ = model(x)
            else:
                mu  = model(x)
                var = torch.ones_like(mu)
            all_mu.append(mu.cpu())
            all_var.append(var.cpu())
            all_y.append(y)

    all_mu  = torch.cat(all_mu)
    all_var = torch.cat(all_var).clamp(min=1e-10)
    all_y   = torch.cat(all_y)

    mae, rmse, smape, mase, crps, nll = compute_metrics(all_mu, all_var, all_y, mad)
    elapsed = round(time.time() - t0, 1)

    model_name = model.__class__.__name__
    print(
        f"  ✓ [{name}]  MAE={mae:.4f}  RMSE={rmse:.4f}  "
        f"sMAPE={smape:.2f}%  MASE={mase:.4f}  "
        f"CRPS={crps:.4f}  NLL={nll:.4f}  "
        f"params={n_par:,}  time={elapsed}s"
    )
    return dict(
        model=model_name, dataset=name,
        mae=round(mae, 4), rmse=round(rmse, 4),
        smape=round(smape, 4), mase=round(mase, 4),
        crps=round(crps, 4) if probabilistic else float("nan"),
        nll=round(nll,  4) if probabilistic else float("nan"),
        n_params=n_par, train_time_sec=elapsed,
    )
