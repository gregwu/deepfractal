"""
DeepFractal end-to-end demo.

Paper: "Stock price dynamics prediction based on multi-scale fractals
        and deep learning" — Du & Tian, PLoS One 20(12):e0335554 (2025)
DOI: 10.1371/journal.pone.0335554

Pipeline:
  1. Fetch S&P 500 closing prices via yfinance (falls back to synthetic)
  2. Extract multi-scale fractal features (windows 16, 32, 64)
  3. Build train/val/test splits (60/20/20)
  4. Train DeepFractal model with multi-scale fractal loss
  5. Evaluate and print MAE / MAPE / MSE / RMSE / R²

Usage:
    cd deepfractal
    python demo.py
    python demo.py --ticker "000001.SS"   # Shanghai Composite
    python demo.py --epochs 50            # quick run
    python demo.py --no-fractal-loss      # ablation: MSE only
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
# Data
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str = "2015-01-01", end: str = "2024-12-31") -> np.ndarray | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if hist.empty:
            return None
        close = hist["Close"].ffill().dropna().values.astype(float)
        print(f"  Loaded {len(close)} trading days from yfinance ({ticker})")
        return close
    except Exception as e:
        print(f"  yfinance failed: {e}")
        return None


def _synthetic_data(n: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)
    close = 3000 * np.exp(np.cumsum(returns))
    print(f"  Using synthetic price series ({n} days)")
    return close


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",          default="^GSPC")
    parser.add_argument("--windows",         default="16,32,64")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--feature-dim",     type=int,   default=64)
    parser.add_argument("--no-fractal-loss", action="store_true", help="Ablation: MSE only")
    parser.add_argument("--no-vae",          action="store_true")
    parser.add_argument("--device",          default="cpu")
    args = parser.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    use_fractal = not args.no_fractal_loss

    print("=" * 60)
    print("  DeepFractal Stock Price Prediction")
    print("  Du & Tian, PLoS One 2025  DOI:10.1371/journal.pone.0335554")
    print("=" * 60)
    print(f"\n  Ticker : {args.ticker}")
    print(f"  Windows: {windows}")
    print(f"  Epochs : {args.epochs}  (patience=20)")
    print(f"  Fractal loss: {'YES' if use_fractal else 'NO (MSE only)'}")

    # 1. Load data
    print("\n[1/5] Loading price data...")
    close = _fetch_yfinance(args.ticker)
    if close is None:
        close = _synthetic_data()

    # 2. Extract multi-scale fractal features
    from fractal_features import extract_multiscale_features
    print(f"\n[2/5] Extracting fractal features at scales {windows}...")
    t0 = time.time()
    scale_feats = extract_multiscale_features(close, windows=tuple(windows))
    elapsed = time.time() - t0
    for w, f in scale_feats.items():
        print(f"  window={w:3d}: {f.shape[0]} samples × {f.shape[1]} features  ({elapsed:.1f}s)")

    # Target: next-day closing price (shift by 1)
    min_len = min(f.shape[0] for f in scale_feats.values())
    targets = np.array([close[-(min_len - i)] for i in range(min_len - 1, -1, -1)])
    # Align: target[t] = close price corresponding to the end of each window
    # We predict the price at the last day of each window
    end_indices = [len(close) - min_len + i for i in range(min_len)]
    targets = close[end_indices]

    # 3. Build dataset
    from model import build_dataset, DeepFractal, train, evaluate
    print("\n[3/5] Building train/val/test splits (60/20/20)...")
    dataset = build_dataset(scale_feats, targets)
    print(f"  Train: {len(dataset['train'])}  Val: {len(dataset['val'])}  Test: {len(dataset['test'])}")

    # 4. Train
    print(f"\n[4/5] Training DeepFractal model...")
    model = DeepFractal(
        fractal_dim=10,
        n_scales=len(windows),
        feature_dim=args.feature_dim,
        tucker_rank=8,
        latent_dim=16,
        dropout=0.3,
        use_vae=not args.no_vae,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    t0 = time.time()
    history = train(
        model, dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=20,
        use_fractal=use_fractal,
        device=args.device,
        verbose=True,
    )
    print(f"  Training time: {time.time()-t0:.1f}s  Best val loss: {history['best_val']:.5f}")

    # 5. Evaluate
    print("\n[5/5] Evaluating on test set...")
    metrics = evaluate(model, dataset, device=args.device)

    print(f"\n{'='*60}")
    print(f"  {'Metric':<8}  {'Value':>12}")
    print(f"  {'-'*38}")
    print(f"  {'MAE':<8}  {metrics['MAE']:>12.4f}")
    print(f"  {'MAPE':<8}  {metrics['MAPE']:>12.4f}")
    print(f"  {'MSE':<8}  {metrics['MSE']:>12.4f}")
    print(f"  {'RMSE':<8}  {metrics['RMSE']:>12.4f}")
    print(f"  {'R²':<8}  {metrics['R2']:>12.4f}")
    print(f"{'='*60}")

    # Paper reference results (Table 1, S&P 500)
    print("\n  Paper Table 1 reference (S&P 500):")
    print(f"  {'Model':<12}  {'MSE':>8}  {'RMSE':>7}  {'MAE':>7}  {'MAPE':>7}  {'R²':>5}")
    print(f"  {'-'*52}")
    ref = [
        ("RNN",      5362.06, 76.35, 58.94, 0.06, 0.99),
        ("LSTM",     4013.28, 68.37, 62.38, 0.05, 0.94),
        ("GRU",      3837.51, 62.76, 49.67, 0.05, 0.95),
        ("ALSTM",    3186.87, 58.39, 16.98, 0.07, 0.92),
        ("VMD-LSTM",  963.85, 43.68, 39.17, 0.03, 0.91),
        ("Proposed",  468.86, 19.63, 15.68, 0.02, 0.86),
    ]
    for name, mse, rmse, mae, mape, r2 in ref:
        marker = " ←" if name == "Proposed" else ""
        print(f"  {name:<12}  {mse:>8.2f}  {rmse:>7.2f}  {mae:>7.2f}  {mape:>7.2f}  {r2:>5.2f}{marker}")
    print()


if __name__ == "__main__":
    main()
