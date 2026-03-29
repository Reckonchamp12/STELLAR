"""
experiments/train_baselines.py
───────────────────────────────
Train and evaluate all baseline models across all 6 datasets in
parallel across up to 2 GPUs.

Models
------
  Point-forecast:   NLinear, DLinear, LSTM, TransformerModel, Informer,
                    PatchTST, TimesNet, FEDformer
  Probabilistic:    DeepAR

Usage
-----
    python experiments/train_baselines.py
    python experiments/train_baselines.py --data_root /path/to/datasets --gpus 2
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.dataset import ALL_DATASETS, set_data_root
from baselines import (
    NLinear, DLinear, LSTM, DeepAR,
    TransformerModel, Informer, PatchTST,
    FEDformer, TimesNet,
)
from experiments.engine import train_and_eval


# ---------------------------------------------------------------------------
# Registry: model_name → (factory, probabilistic)
# ---------------------------------------------------------------------------
MODELS: dict[str, tuple] = {
    "NLinear":     (lambda: NLinear(),          False),
    "DLinear":     (lambda: DLinear(),          False),
    "LSTM":        (lambda: LSTM(),             False),
    "DeepAR":      (lambda: DeepAR(),           True),
    "Transformer": (lambda: TransformerModel(), False),
    "Informer":    (lambda: Informer(),         False),
    "PatchTST":    (lambda: PatchTST(),         False),
    "FEDformer":   (lambda: FEDformer(),        False),
    "TimesNet":    (lambda: TimesNet(),         False),
}


def main(args: argparse.Namespace) -> None:
    if args.data_root:
        set_data_root(args.data_root)

    model_names = args.models or list(MODELS.keys())
    datasets    = args.datasets or ALL_DATASETS
    n_gpus      = args.gpus
    all_results, all_errors = [], {}

    for mname in model_names:
        factory, prob = MODELS[mname]
        print(f"\n{'='*60}\nBaseline: {mname}\n{'='*60}")
        gpu_map = {ds: i % n_gpus for i, ds in enumerate(datasets)}

        with ThreadPoolExecutor(max_workers=n_gpus) as ex:
            futures = {
                ex.submit(
                    train_and_eval, ds, gpu_map[ds], factory,
                    seq_len=args.seq_len, pred_len=args.pred_len,
                    batch=args.batch, lr=args.lr, epochs=args.epochs,
                    patience=args.patience, probabilistic=prob,
                ): ds
                for ds in datasets
            }
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    r = fut.result()
                    r["model"] = mname
                    all_results.append(r)
                except Exception as e:
                    print(f"  [ERROR] {mname}/{ds}: {e}")
                    all_errors[f"{mname}/{ds}"] = str(e)

    df = pd.DataFrame(all_results).sort_values(["model", "dataset"]).reset_index(drop=True)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/baseline_results.csv", index=False)

    print("\n" + "=" * 60)
    print(" BASELINE FINAL RESULTS")
    print("=" * 60)
    print(df[["model", "dataset", "mae", "rmse", "smape", "mase", "crps", "nll",
              "n_params", "train_time_sec"]].to_string(index=False))

    if all_errors:
        print(f"\nErrors: {all_errors}")
    print("\nSaved → results/baseline_results.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Train all baselines")
    p.add_argument("--data_root", default=None)
    p.add_argument("--models",    nargs="+", default=None, choices=list(MODELS.keys()),
                   help="Subset of models to run (default: all)")
    p.add_argument("--datasets",  nargs="+", default=None, choices=ALL_DATASETS)
    p.add_argument("--gpus",      type=int, default=2)
    p.add_argument("--seq_len",   type=int, default=96)
    p.add_argument("--pred_len",  type=int, default=24)
    p.add_argument("--batch",     type=int, default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--patience",  type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
