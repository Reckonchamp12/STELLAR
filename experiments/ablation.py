"""
experiments/ablation.py
────────────────────────
Five ablation variants of STELLAR.

Variants
--------
  STELLAR          — full model (baseline)
  NoKoopman        — Koopman path replaced by zeros
  NoAdaptiveFilter — bypass ASF, use raw normalised input
  NoMRTP           — replace MRTP with a simple linear projection
  NoGating         — fixed equal weights (1/3 each) instead of learned gate
  NoNPHead         — NP head replaced by a deterministic head (no KL)

Usage
-----
    python experiments/ablation.py
    python experiments/ablation.py --data_root /path/to/datasets --gpus 2
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.dataset import ALL_DATASETS, set_data_root
from stellar.model import STELLAR
from stellar.components import NPHead
from experiments.engine import train_and_eval


# ---------------------------------------------------------------------------
# Ablation subclasses
# ---------------------------------------------------------------------------

class STELLAR_NoKoopman(STELLAR):
    """Koopman path disabled (output zeroed)."""
    def forward(self, x):
        xn   = self.revin.norm(x)
        x_sp = self.asf(xn)
        f_sp = self.asf_proj(x_sp)
        koop = torch.zeros(xn.size(0), self.pred_len, device=x.device)
        f_mr = self.mrtp(xn)
        lin  = self.linear(xn)
        ctx  = torch.cat([f_sp, f_mr], dim=-1)
        g    = self.gate(ctx)
        g_k, g_l, g_np = g[:, 0:1], g[:, 1:2], g[:, 2:3]
        np_mu, np_var, kl = self.nphead(ctx)
        mean_n = g_k * koop + g_l * lin + g_np * np_mu
        mean   = self.revin.denorm(mean_n)
        var    = np_var * self.revin._sig ** 2
        return mean, var, kl


class STELLAR_NoAdaptiveFilter(STELLAR):
    """ASF bypassed — raw normalised input fed to projection."""
    def forward(self, x):
        xn   = self.revin.norm(x)
        f_sp = self.asf_proj(xn)                  # skip self.asf
        koop = self.koopman(xn)
        f_mr = self.mrtp(xn)
        lin  = self.linear(xn)
        ctx  = torch.cat([f_sp, f_mr], dim=-1)
        g    = self.gate(ctx)
        g_k, g_l, g_np = g[:, 0:1], g[:, 1:2], g[:, 2:3]
        np_mu, np_var, kl = self.nphead(ctx)
        mean_n = g_k * koop + g_l * lin + g_np * np_mu
        mean   = self.revin.denorm(mean_n)
        var    = np_var * self.revin._sig ** 2
        return mean, var, kl


class STELLAR_NoMRTP(STELLAR):
    """MRTP replaced by a single linear projection."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_out = self.mrtp.proj.out_features
        self.simple_proj = nn.Linear(96, d_out)  # seq_len default

    def forward(self, x):
        xn   = self.revin.norm(x)
        x_sp = self.asf(xn)
        f_sp = self.asf_proj(x_sp)
        koop = self.koopman(xn)
        f_mr = self.simple_proj(xn)              # replace MRTP
        lin  = self.linear(xn)
        ctx  = torch.cat([f_sp, f_mr], dim=-1)
        g    = self.gate(ctx)
        g_k, g_l, g_np = g[:, 0:1], g[:, 1:2], g[:, 2:3]
        np_mu, np_var, kl = self.nphead(ctx)
        mean_n = g_k * koop + g_l * lin + g_np * np_mu
        mean   = self.revin.denorm(mean_n)
        var    = np_var * self.revin._sig ** 2
        return mean, var, kl


class STELLAR_NoGating(STELLAR):
    """Fixed equal weights (1/3 each) instead of learned gate."""
    def forward(self, x):
        xn   = self.revin.norm(x)
        x_sp = self.asf(xn)
        f_sp = self.asf_proj(x_sp)
        koop = self.koopman(xn)
        f_mr = self.mrtp(xn)
        lin  = self.linear(xn)
        ctx  = torch.cat([f_sp, f_mr], dim=-1)
        B    = xn.size(0)
        g_k  = torch.full((B, 1), 1/3, device=x.device)
        g_l  = torch.full((B, 1), 1/3, device=x.device)
        g_np = torch.full((B, 1), 1/3, device=x.device)
        np_mu, np_var, kl = self.nphead(ctx)
        mean_n = g_k * koop + g_l * lin + g_np * np_mu
        mean   = self.revin.denorm(mean_n)
        var    = np_var * self.revin._sig ** 2
        return mean, var, kl


class STELLAR_NoNPHead(STELLAR):
    """
    NP head replaced by a deterministic head.
    Uses the NP decoder with z=0 (no sampling, no KL penalty).
    A separate variance head is attached for fair probabilistic comparison.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_ctx = 64 * 2
        self.det_var_head = nn.Linear(d_ctx, self.pred_len)

    def forward(self, x):
        xn   = self.revin.norm(x)
        x_sp = self.asf(xn)
        f_sp = self.asf_proj(x_sp)
        koop = self.koopman(xn)
        f_mr = self.mrtp(xn)
        lin  = self.linear(xn)
        ctx  = torch.cat([f_sp, f_mr], dim=-1)

        # deterministic decode: z = 0
        fake_z = torch.zeros(ctx.size(0), self.nphead.d_lat, device=x.device)
        h      = self.nphead.dec(torch.cat([ctx, fake_z], dim=-1))
        np_mu  = self.nphead.mu_h(h)
        np_var = F.softplus(self.det_var_head(ctx)) + 1e-6
        kl     = torch.tensor(0.0, device=x.device)

        g = self.gate(ctx)
        g_k, g_l, g_np = g[:, 0:1], g[:, 1:2], g[:, 2:3]
        mean_n = g_k * koop + g_l * lin + g_np * np_mu
        mean   = self.revin.denorm(mean_n)
        var    = np_var * self.revin._sig ** 2
        return mean, var, kl


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
VARIANTS = {
    "STELLAR":           STELLAR,
    "NoKoopman":         STELLAR_NoKoopman,
    "NoAdaptiveFilter":  STELLAR_NoAdaptiveFilter,
    "NoMRTP":            STELLAR_NoMRTP,
    "NoGating":          STELLAR_NoGating,
    "NoNPHead":          STELLAR_NoNPHead,
}


def main(args: argparse.Namespace) -> None:
    if args.data_root:
        set_data_root(args.data_root)

    datasets = args.datasets or ALL_DATASETS
    n_gpus   = args.gpus
    all_results = []

    for vname, VModel in VARIANTS.items():
        print(f"\n{'='*60}\nAblation: {vname}\n{'='*60}")
        gpu_map = {ds: i % n_gpus for i, ds in enumerate(datasets)}
        with ThreadPoolExecutor(max_workers=n_gpus) as ex:
            futures = {
                ex.submit(
                    train_and_eval, ds, gpu_map[ds], VModel,
                    seq_len=args.seq_len, pred_len=args.pred_len,
                    batch=args.batch, lr=args.lr, epochs=args.epochs,
                    patience=args.patience, kl_weight=args.kl_weight,
                    probabilistic=True,
                ): ds
                for ds in datasets
            }
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    r = fut.result()
                    r["variant"] = vname
                    all_results.append(r)
                except Exception as e:
                    print(f"  [ERROR] {vname}/{ds}: {e}")

    df = pd.DataFrame(all_results).sort_values(["variant", "dataset"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ablation_results.csv", index=False)
    print("\n\nAblation results saved → results/ablation_results.csv")

    # Summary pivot
    pivot = df.pivot_table(index="dataset", columns="variant", values="mae")
    print("\nMAE by variant:")
    print(pivot.round(4).to_string())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None)
    p.add_argument("--datasets",  nargs="+", default=None, choices=ALL_DATASETS)
    p.add_argument("--gpus",      type=int, default=2)
    p.add_argument("--seq_len",   type=int, default=96)
    p.add_argument("--pred_len",  type=int, default=24)
    p.add_argument("--batch",     type=int, default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--epochs",    type=int, default=20)
    p.add_argument("--patience",  type=int, default=5)
    p.add_argument("--kl_weight", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
