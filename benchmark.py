"""
Full benchmark: all models from Table 1 of the paper.

Models compared:
  RNN, LSTM, GRU, ALSTM, VMD-LSTM  (baselines)
  DeepFractal                        (proposed)

Evaluation metrics: MAE, MAPE, MSE, RMSE, R²  (Eqs. 27-31)

Usage:
    cd deepfractal
    python benchmark.py                     # S&P 500
    python benchmark.py --ticker "^GSPC"
    python benchmark.py --epochs 100        # faster
    python benchmark.py --seq-len 30
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch(ticker: str) -> np.ndarray | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(start="2015-01-01", end="2024-12-31")
        if hist.empty:
            return None
        return hist["Close"].ffill().dropna().values.astype(float)
    except Exception:
        return None


def _synthetic(n: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 3000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def _print_row(name: str, m: dict, best: dict, elapsed: float) -> None:
    def _mark(key):
        return "✓" if abs(m[key] - best[key]) < 1e-6 else " "
    print(
        f"  {name:<12} "
        f"{m['MSE']:>12,.1f} "
        f"{m['RMSE']:>8.2f}{_mark('RMSE')} "
        f"{m['MAE']:>8.2f}{_mark('MAE')} "
        f"{m['MAPE']:>7.4f}{_mark('MAPE')} "
        f"{m['R2']:>7.4f}  "
        f"{elapsed:>5.1f}s"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="^GSPC")
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--seq-len",  type=int, default=60)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--windows",  default="16,32,64")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    windows = [int(w) for w in args.windows.split(",")]

    print("=" * 75)
    print("  DeepFractal Benchmark — all models from Table 1")
    print(f"  Ticker: {args.ticker}  |  seq_len={args.seq_len}  |  epochs={args.epochs}")
    print("=" * 75)

    # ---- Load data ----
    print(f"\nLoading {args.ticker} ...")
    close = _fetch(args.ticker)
    if close is None:
        print("  yfinance unavailable — using synthetic data")
        close = _synthetic()
    else:
        print(f"  {len(close)} trading days (2015-2024)")

    # ---- Sequence dataset for baselines ----
    from baselines import (
        make_sequence_dataset,
        RNNModel, LSTMModel, GRUModel, ALSTMModel, VMDLSTMModel,
        train_baseline,
    )
    seq_ds = make_sequence_dataset(close, seq_len=args.seq_len)
    print(f"  Sequence dataset: train={seq_ds['n_train']}  val={seq_ds['n_val']}  test={seq_ds['n_test']}")

    # ---- Fractal dataset for DeepFractal ----
    from fractal_features import extract_multiscale_features
    from model import build_dataset, DeepFractal, train, evaluate

    print(f"\nExtracting fractal features at scales {windows}...")
    t0 = time.time()
    scale_feats = extract_multiscale_features(close, windows=tuple(windows))
    print(f"  Done in {time.time()-t0:.1f}s")

    min_len = min(f.shape[0] for f in scale_feats.values())
    end_idx = [len(close) - min_len + i for i in range(min_len)]
    targets = close[end_idx]
    frac_ds = build_dataset(scale_feats, targets)
    print(f"  Fractal dataset:  train={len(frac_ds['train'])}  val={len(frac_ds['val'])}  test={len(frac_ds['test'])}")

    # ---- Run all models ----
    all_results = {}
    h = args.hidden

    specs = [
        ("RNN",      RNNModel(hidden=h),      False),
        ("LSTM",     LSTMModel(hidden=h),     False),
        ("GRU",      GRUModel(hidden=h),      False),
        ("ALSTM",    ALSTMModel(hidden=h),    False),
        ("VMD-LSTM", VMDLSTMModel(K=3, hidden=h), True),
    ]

    print(f"\nTraining baselines (epochs={args.epochs}, lr={args.lr}) ...\n")
    for name, model, is_vmd in specs:
        print(f"  [{name}] training...", end=" ", flush=True)
        t0 = time.time()
        metrics = train_baseline(
            model, seq_ds,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            patience=20,
            device=args.device,
            seed=args.seed,
            is_vmd=is_vmd,
        )
        elapsed = time.time() - t0
        all_results[name] = (metrics, elapsed)
        print(f"done ({elapsed:.1f}s)  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R²={metrics['R2']:.4f}")

    # ---- DeepFractal ----
    print(f"\n  [DeepFractal] training...", end=" ", flush=True)
    t0 = time.time()
    df_model = DeepFractal(
        fractal_dim=10, n_scales=len(windows),
        feature_dim=64, tucker_rank=8,
        latent_dim=16, dropout=0.3, use_vae=True,
    )
    train(df_model, frac_ds, epochs=args.epochs, batch_size=args.batch,
          lr=args.lr, patience=20, use_fractal=True, device=args.device,
          seed=args.seed, verbose=False)
    elapsed = time.time() - t0
    df_metrics = evaluate(df_model, frac_ds, device=args.device)
    all_results["DeepFractal"] = (df_metrics, elapsed)
    print(f"done ({elapsed:.1f}s)  RMSE={df_metrics['RMSE']:.2f}  MAE={df_metrics['MAE']:.2f}  R²={df_metrics['R2']:.4f}")

    # ---- Results table ----
    best = {
        "MSE":  min(m["MSE"]  for m, _ in all_results.values()),
        "RMSE": min(m["RMSE"] for m, _ in all_results.values()),
        "MAE":  min(m["MAE"]  for m, _ in all_results.values()),
        "MAPE": min(m["MAPE"] for m, _ in all_results.values()),
        "R2":   max(m["R2"]   for m, _ in all_results.values()),
    }

    print("\n" + "=" * 75)
    print(f"  {'Model':<12} {'MSE':>12}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>7}  {'R²':>7}  {'Time':>6}")
    print("  " + "-" * 71)
    order = ["RNN", "LSTM", "GRU", "ALSTM", "VMD-LSTM", "DeepFractal"]
    for name in order:
        m, el = all_results[name]
        _print_row(name, m, best, el)
    print("=" * 75)
    print("  ✓ = best value in column")

    # ---- Paper reference ----
    print("\n  Paper Table 1 reference (S&P 500, Chinese market data):")
    print(f"  {'Model':<12} {'MSE':>12}  {'RMSE':>9}  {'MAE':>8}  {'MAPE':>7}  {'R²':>7}")
    print("  " + "-" * 63)
    paper = [
        ("RNN",      5362.06, 76.35, 58.94, 0.06, 0.99),
        ("LSTM",     4013.28, 68.37, 62.38, 0.05, 0.94),
        ("GRU",      3837.51, 62.76, 49.67, 0.05, 0.95),
        ("ALSTM",    3186.87, 58.39, 16.98, 0.07, 0.92),
        ("VMD-LSTM",  963.85, 43.68, 39.17, 0.03, 0.91),
        ("DeepFractal", 468.86, 19.63, 15.68, 0.02, 0.86),
    ]
    for name, mse, rmse, mae, mape, r2 in paper:
        marker = " ←" if name == "DeepFractal" else ""
        print(f"  {name:<12} {mse:>12,.2f}  {rmse:>9.2f}  {mae:>8.2f}  {mape:>7.2f}  {r2:>7.2f}{marker}")
    print()


if __name__ == "__main__":
    main()
