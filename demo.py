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
import os
import psycopg2
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env
load_dotenv()


# ---------------------------------------------------------------------------
# Database Connection
# ---------------------------------------------------------------------------

def get_db_connection():
    """Connect to PostgreSQL database using .env configuration."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "dbname"),
            user=os.getenv("DB_USER", "dbuser"),
            password=os.getenv("DB_PASSWORD", "dbpassword"),
        )
        return conn
    except psycopg2.Error as e:
        print(f"  Database connection failed: {e}")
        print("  Proceeding without database storage...")
        return None


def save_results_to_db(conn, ticker: str, metrics: dict, model_params: dict):
    """Save training results to database."""
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO deepfractal_results (ticker, mae, mape, mse, rmse, r2, epochs, batch_size, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                mae = EXCLUDED.mae,
                mape = EXCLUDED.mape,
                mse = EXCLUDED.mse,
                rmse = EXCLUDED.rmse,
                r2 = EXCLUDED.r2,
                epochs = EXCLUDED.epochs,
                batch_size = EXCLUDED.batch_size,
                timestamp = EXCLUDED.timestamp
        """, (
            ticker,
            metrics['MAE'],
            metrics['MAPE'],
            metrics['MSE'],
            metrics['RMSE'],
            metrics['R2'],
            model_params.get('epochs', 200),
            model_params.get('batch_size', 64),
            datetime.now(),
        ))
        conn.commit()
        print(f"  Results saved to database for {ticker}")
    except psycopg2.Error as e:
        print(f"  Error saving to database: {e}")


# ---------------------------------------------------------------------------
# S&P 500 Tickers
# ---------------------------------------------------------------------------

def get_sp500_tickers(conn, n_tickers: int = 500) -> list[str]:
    """Get up to n_tickers from stock_data that have sufficient history."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticker
        FROM stock_data
        WHERE date >= '2020-01-01'
        GROUP BY ticker
        HAVING COUNT(*) >= 100
        ORDER BY COUNT(*) DESC, ticker
        LIMIT %s
    """, (n_tickers,))
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    print(f"  Found {len(tickers)} tickers with 100+ days of data in stock_db")
    return tickers


def _fetch_ohlcv(conn, ticker: str, start: str, end: str):
    """Returns (close, ohlcv_5col) or (None, None). ohlcv cols: open,high,low,close,log_vol."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT open_price, high_price, low_price, close_price, volume
            FROM stock_data
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date ASC
        """, (ticker, start, end))
        rows = cursor.fetchall()
        cursor.close()
        if not rows:
            return None, None
        mat = np.array(rows, dtype=float)
        for i in range(1, len(mat)):
            if np.isnan(mat[i]).any():
                mat[i] = mat[i - 1]
        mat = mat[~np.isnan(mat).any(axis=1)]
        if len(mat) < 100:
            return None, None
        close = mat[:, 3]
        ohlcv = mat.copy()
        ohlcv[:, 4] = np.log1p(ohlcv[:, 4])
        print(f"  Loaded {len(close)} trading days from stock_db ({ticker})")
        return close, ohlcv
    except Exception as e:
        print(f"  DB fetch failed for {ticker}: {e}")
        return None, None


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
    parser.add_argument("--sp500",           type=int, default=0, help="Train on N S&P 500 tickers (default: 0=single ticker)")
    parser.add_argument("--windows",         default="16,32,64")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--feature-dim",     type=int,   default=64)
    parser.add_argument("--no-fractal-loss", action="store_true", help="Ablation: MSE only")
    parser.add_argument("--no-vae",          action="store_true")
    parser.add_argument("--device",          default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--data-start",      default="2020-01-01", help="Data start date (YYYY-MM-DD)")
    parser.add_argument("--data-end",        default="2024-12-31", help="Data end date (YYYY-MM-DD)")
    args = parser.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    use_fractal = not args.no_fractal_loss

    # Connect to database
    db_conn = get_db_connection()

    # Load SPY as market reference
    spy_close, _ = _fetch_ohlcv(db_conn, "SPY.US", args.data_start, args.data_end)
    if spy_close is None:
        print("  WARNING: SPY.US not found — SPY ratio feature disabled")

    # Get list of tickers to train on
    if args.sp500 > 0:
        tickers = get_sp500_tickers(db_conn, n_tickers=min(args.sp500, 500))
        print(f"  Training on {len(tickers)} tickers from stock_db")
    else:
        tickers = [args.ticker]

    print("=" * 60)
    print("  DeepFractal Stock Price Prediction")
    print("  Du & Tian, PLoS One 2025  DOI:10.1371/journal.pone.0335554")
    print("=" * 60)
    print(f"\n  Tickers: {len(tickers)} (first: {tickers[0]})")
    print(f"  Data range: {args.data_start} to {args.data_end}")
    print(f"  Windows: {windows}")
    print(f"  Epochs : {args.epochs}  (patience=20)")
    print(f"  Fractal loss: {'YES' if use_fractal else 'NO (MSE only)'}")

    # Results aggregation
    all_metrics = {}
    failed_tickers = []

    # Train on each ticker
    for idx, ticker in enumerate(tickers):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(tickers)}] Training on {ticker}")
        print(f"{'='*60}")

        # 1. Load data
        print(f"\n[1/5] Loading price data...")
        close, ohlcv = _fetch_ohlcv(db_conn, ticker, start=args.data_start, end=args.data_end)
        if close is None:
            print(f"  Failed to load data for {ticker}")
            failed_tickers.append(ticker)
            continue

        if len(close) < 100:
            print(f"  Insufficient data for {ticker} ({len(close)} days < 100)")
            failed_tickers.append(ticker)
            continue

        # Align SPY to same length
        spy = spy_close[-len(close):] if spy_close is not None else None
        if spy is not None and len(spy) < len(close):
            spy = np.pad(spy, (len(close) - len(spy), 0), mode='edge')

        # 2. Extract multi-scale fractal features
        from fractal_features import extract_multiscale_features, append_ohlcv_spy
        print(f"\n[2/5] Extracting fractal features at scales {windows}...")
        try:
            t0 = time.time()
            scale_feats = extract_multiscale_features(close, windows=tuple(windows))
            if spy is not None:
                scale_feats = append_ohlcv_spy(scale_feats, ohlcv, spy)
            elapsed = time.time() - t0
            for w, f in scale_feats.items():
                print(f"  window={w:3d}: {f.shape[0]} samples × {f.shape[1]} features  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  Feature extraction failed for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

        min_len = min(f.shape[0] for f in scale_feats.values())
        end_indices = [len(close) - min_len + i for i in range(min_len)]
        targets = close[end_indices]
        aligned = {w: scale_feats[w][-min_len:] for w in windows}

        # 3. Build dataset
        from model import build_dataset, DeepFractal, train, evaluate
        print("\n[3/5] Building train/val/test splits (60/20/20)...")
        try:
            dataset = build_dataset(aligned, targets)
            print(f"  Train: {len(dataset['train'])}  Val: {len(dataset['val'])}  Test: {len(dataset['test'])}")
        except Exception as e:
            print(f"  Dataset building failed for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

        fractal_dim = scale_feats[windows[0]].shape[1]  # 10 or 16

        # 4. Train
        print(f"\n[4/5] Training DeepFractal model...")
        try:
            model = DeepFractal(
                fractal_dim=fractal_dim,
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
                verbose=False,  # Reduce verbosity for multiple tickers
            )
            print(f"  Training time: {time.time()-t0:.1f}s  Best val loss: {history['best_val']:.5f}")
        except Exception as e:
            print(f"  Training failed for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

        # 5. Evaluate
        print(f"\n[5/5] Evaluating on test set...")
        try:
            metrics = evaluate(model, dataset, device=args.device)

            print(f"\n  {'Metric':<8}  {'Value':>12}")
            print(f"  {'-'*24}")
            print(f"  {'MAE':<8}  {metrics['MAE']:>12.4f}")
            print(f"  {'MAPE':<8}  {metrics['MAPE']:>12.4f}")
            print(f"  {'MSE':<8}  {metrics['MSE']:>12.4f}")
            print(f"  {'RMSE':<8}  {metrics['RMSE']:>12.4f}")
            print(f"  {'R²':<8}  {metrics['R2']:>12.4f}")

            all_metrics[ticker] = metrics

            # Save to database
            save_results_to_db(db_conn, ticker, metrics, {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            })
        except Exception as e:
            print(f"  Evaluation failed for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

        # Stop after first ticker if only one ticker requested
        if len(tickers) == 1:
            break

    # Summary
    if len(tickers) > 1:
        print(f"\n\n{'='*60}")
        print(f"  SUMMARY: Trained on {len(all_metrics)}/{len(tickers)} tickers")
        print(f"{'='*60}")
        if failed_tickers:
            print(f"\n  Failed tickers ({len(failed_tickers)}):")
            for t in failed_tickers[:10]:
                print(f"    - {t}")
            if len(failed_tickers) > 10:
                print(f"    ... and {len(failed_tickers)-10} more")

        if all_metrics:
            mae_vals = [m['MAE'] for m in all_metrics.values()]
            rmse_vals = [m['RMSE'] for m in all_metrics.values()]
            r2_vals = [m['R2'] for m in all_metrics.values()]

            print(f"\n  Aggregated Results (across {len(all_metrics)} tickers):")
            print(f"  {'Metric':<8}  {'Mean':>12}  {'Std':>12}  {'Min':>12}  {'Max':>12}")
            print(f"  {'-'*60}")
            print(f"  {'MAE':<8}  {np.mean(mae_vals):>12.4f}  {np.std(mae_vals):>12.4f}  {np.min(mae_vals):>12.4f}  {np.max(mae_vals):>12.4f}")
            print(f"  {'RMSE':<8}  {np.mean(rmse_vals):>12.4f}  {np.std(rmse_vals):>12.4f}  {np.min(rmse_vals):>12.4f}  {np.max(rmse_vals):>12.4f}")
            print(f"  {'R²':<8}  {np.mean(r2_vals):>12.4f}  {np.std(r2_vals):>12.4f}  {np.min(r2_vals):>12.4f}  {np.max(r2_vals):>12.4f}")
            print(f"{'='*60}\n")

    if db_conn:
        db_conn.close()



if __name__ == "__main__":
    main()
