"""
experiments/train_stellar.py
─────────────────────────────
Train and evaluate STELLAR on all 6 standard datasets in parallel across
up to 2 GPUs.

Usage
-----
    python experiments/train_stellar.py
    python experiments/train_stellar.py --data_root /path/to/datasets --gpus 2
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.dataset import ALL_DATASETS, set_data_root
from stellar import STELLAR
from experiments.engine import train_and_eval


# ---------------------------------------------------------------------------
# Baseline MAE values for comparison
# ---------------------------------------------------------------------------
BEST_BASELINE_MAE = {
    "ECL": 0.2512, "ETTh1": 0.4685, "ETTh2": 0.3210,
    "ETTm1": 0.3794, "ETTm2": 0.2564, "Weather": 0.0468,
}


def main(args: argparse.Namespace) -> None:
    if args.data_root:
        set_data_root(args.data_root)

    datasets = args.datasets or ALL_DATASETS
    n_gpus   = args.gpus
    gpu_map  = {ds: i % n_gpus for i, ds in enumerate(datasets)}

    results, errors = [], {}

    print("=" * 60)
    print(" STELLAR — starting parallel training")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futures = {
            ex.submit(
                train_and_eval,
                ds,
                gpu_map[ds],
                lambda: STELLAR(
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    n_modes=args.n_modes,
                ),
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                batch=args.batch,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                kl_weight=args.kl_weight,
                probabilistic=True,
            ): ds
            for ds in datasets
        }
        for fut in as_completed(futures):
            ds = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] {ds}: {e}")
                errors[ds] = str(e)

    df = pd.DataFrame(results).sort_values("dataset").reset_index(drop=True)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/stellar_results.csv", index=False)

    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n── MAE improvement vs best baseline ──")
    for _, row in df.iterrows():
        base = BEST_BASELINE_MAE.get(row["dataset"])
        if base:
            delta = (base - row["mae"]) / base * 100
            sign  = "✓" if delta > 0 else "✗"
            print(f"  {sign} {row['dataset']:8s}  STELLAR={row['mae']:.4f}  "
                  f"base={base:.4f}  Δ={delta:+.1f}%")

    if errors:
        print(f"\nErrors encountered: {errors}")
    print("\nSaved → results/stellar_results.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STELLAR on 6 benchmarks")
    p.add_argument("--data_root", default=None,
                   help="Root directory containing dataset folders")
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=ALL_DATASETS,
                   help="Datasets to run (default: all 6)")
    p.add_argument("--gpus",      type=int, default=2)
    p.add_argument("--seq_len",   type=int, default=96)
    p.add_argument("--pred_len",  type=int, default=24)
    p.add_argument("--batch",     type=int, default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--epochs",    type=int, default=20)
    p.add_argument("--patience",  type=int, default=5)
    p.add_argument("--kl_weight", type=float, default=1e-3)
    p.add_argument("--n_modes",   type=int, default=32,
                   help="Number of Koopman spectral modes")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
