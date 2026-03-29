"""
data/dataset.py
───────────────
Shared dataset utilities used by STELLAR and all baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Default paths (Kaggle layout).  Override via CLI --data_root.
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("/kaggle/input/datasets/rahuldray12324/six-datasets/all_six_datasets")

DATASET_PATHS: dict[str, Path] = {
    "ETTh1":   DEFAULT_ROOT / "ETTh1"   / "Y_df.csv",
    "ETTh2":   DEFAULT_ROOT / "ETTh2"   / "Y_df.csv",
    "ETTm1":   DEFAULT_ROOT / "ETTm1"   / "Y_df.csv",
    "ETTm2":   DEFAULT_ROOT / "ETTm2"   / "Y_df.csv",
    "Weather": DEFAULT_ROOT / "Weather" / "Y_df.csv",
    "ECL":     DEFAULT_ROOT / "ECL"     / "Y_df.csv",
}

ALL_DATASETS = list(DATASET_PATHS.keys())


def set_data_root(root: str | Path) -> None:
    """Update all paths to use a custom root directory."""
    root = Path(root)
    for name in DATASET_PATHS:
        DATASET_PATHS[name] = root / name / "Y_df.csv"


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class TSDataset(Dataset):
    """Sliding-window univariate time-series dataset."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int) -> None:
        self.x = torch.tensor(data, dtype=torch.float32)
        self.sl = seq_len
        self.pl = pred_len

    def __len__(self) -> int:
        return len(self.x) - self.sl - self.pl + 1

    def __getitem__(self, i: int):
        return (
            self.x[i : i + self.sl],
            self.x[i + self.sl : i + self.sl + self.pl],
        )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _detect_target(df: pd.DataFrame) -> str:
    """Return the name of the first usable numeric column."""
    if "y" in df.columns:
        return "y"
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    raise ValueError("No numeric column found in DataFrame.")


def load_dataset(
    name: str,
    seq_len: int = 96,
    pred_len: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> tuple[TSDataset, TSDataset, TSDataset, StandardScaler, float]:
    """
    Load, split (70/10/20 chronological), scale, and return PyTorch datasets.

    Returns
    -------
    train_ds, val_ds, test_ds : TSDataset
    scaler                    : fitted StandardScaler
    mad                       : mean absolute diff on training set (MASE denom)
    """
    path = DATASET_PATHS[name]
    df = pd.read_csv(path)
    target_col = _detect_target(df)
    vals = df[target_col].values.astype(np.float32)

    n = len(vals)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    sc = StandardScaler()
    tr = sc.fit_transform(vals[:t1].reshape(-1, 1)).flatten()
    va = sc.transform(vals[t1:t2].reshape(-1, 1)).flatten()
    te = sc.transform(vals[t2:].reshape(-1, 1)).flatten()

    # Naive seasonal MAD (lag-1) for MASE denominator
    mad = float(np.mean(np.abs(np.diff(tr))))

    return (
        TSDataset(tr, seq_len, pred_len),
        TSDataset(va, seq_len, pred_len),
        TSDataset(te, seq_len, pred_len),
        sc,
        mad,
    )


def make_loader(
    ds: TSDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
