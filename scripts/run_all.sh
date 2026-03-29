#!/usr/bin/env bash
# scripts/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Full reproduction pipeline.
#
# Usage:
#   bash scripts/run_all.sh /path/to/all_six_datasets
#
# The script runs:
#   1. Baselines benchmark   → results/baseline_results.csv
#   2. STELLAR benchmark     → results/stellar_results.csv
#   3. Ablation study        → results/ablation_results.csv
#   4. Hyperparameter sweep  → results/hyperparam_sensitivity.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_ROOT="${1:-/kaggle/input/datasets/rahuldray12324/six-datasets/all_six_datasets}"
GPUS="${2:-2}"

echo "═══════════════════════════════════════════════════════════"
echo " STELLAR — Full Reproduction Pipeline"
echo " DATA_ROOT : $DATA_ROOT"
echo " GPUs      : $GPUS"
echo "═══════════════════════════════════════════════════════════"

mkdir -p results

# ── Step 1: Baselines ────────────────────────────────────────────────────────
echo ""
echo "Step 1/4: Running baselines..."
python experiments/train_baselines.py \
    --data_root "$DATA_ROOT" \
    --gpus "$GPUS" \
    --epochs 10 \
    --patience 3

# ── Step 2: STELLAR ──────────────────────────────────────────────────────────
echo ""
echo "Step 2/4: Training STELLAR..."
python experiments/train_stellar.py \
    --data_root "$DATA_ROOT" \
    --gpus "$GPUS" \
    --epochs 20 \
    --patience 5

# ── Step 3: Ablation ─────────────────────────────────────────────────────────
echo ""
echo "Step 3/4: Running ablation study..."
python experiments/ablation.py \
    --data_root "$DATA_ROOT" \
    --gpus "$GPUS" \
    --epochs 20 \
    --patience 5

# ── Step 4: Hyperparameter sweep ─────────────────────────────────────────────
echo ""
echo "Step 4/4: Running hyperparameter sweep..."
python experiments/hyperparam_sweep.py \
    --data_root "$DATA_ROOT" \
    --datasets Weather ECL \
    --n_modes 16 32 64 128 \
    --epochs 20 \
    --patience 5

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " All done! Results:"
ls -lh results/*.csv
echo "═══════════════════════════════════════════════════════════"
