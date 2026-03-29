"""
experiments/hyperparam_sweep.py
────────────────────────────────
Koopman n_modes sensitivity sweep on Weather and ECL.

Usage
-----
    python experiments/hyperparam_sweep.py
    python experiments/hyperparam_sweep.py --datasets Weather ECL --n_modes 16 32 64 128
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from stellar.model import STELLAR
from data.dataset import set_data_root
from experiments.engine import train_and_eval


def main(args: argparse.Namespace) -> None:
    if args.data_root:
        set_data_root(args.data_root)

    results = []
    for ds in args.datasets:
        print(f"\nHyperparameter sweep on {ds}")
        for modes in args.n_modes:
            print(f"\n  n_modes = {modes}")
            r = train_and_eval(
                ds,
                gpu_id=0,
                model_factory=lambda m=modes: STELLAR(n_modes=m),
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                batch=args.batch,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                kl_weight=args.kl_weight,
                probabilistic=True,
            )
            r["n_modes"] = modes
            results.append(r)

    df = pd.DataFrame(results).sort_values(["dataset", "n_modes"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/hyperparam_sensitivity.csv", index=False)
    print("\nHyperparameter sensitivity results saved → results/hyperparam_sensitivity.csv")
    print(df[["dataset", "n_modes", "mae", "rmse", "crps"]].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None)
    p.add_argument("--datasets",  nargs="+", default=["Weather", "ECL"])
    p.add_argument("--n_modes",   nargs="+", type=int, default=[16, 32, 64, 128])
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
