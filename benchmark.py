"""
Full benchmark: all models from Table 1 of the paper.

Models compared:
  RNN, LSTM, GRU, ALSTM, VMD-LSTM  (baselines)
  DeepFractal                        (proposed)

Evaluation metrics: MAE, MAPE, MSE, RMSE, R²  (Eqs. 27-31)

Paper setup: trained on ~3000 stocks, metrics aggregated across stocks.
This reproduction uses US stocks from stock_db (2015-2024), one model
trained and evaluated per stock, results averaged across all stocks.

Usage:
    ./run.sh benchmark.py                        # 3000 stocks, 200 epochs
    ./run.sh benchmark.py --n-stocks 500         # faster subset
    ./run.sh benchmark.py --epochs 100
    ./run.sh benchmark.py --seq-len 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "stock_db"),
        user=os.getenv("DB_USER", "gangwu"),
        password=os.getenv("DB_PASSWORD", "gangwuspark"),
    )


def _load_tickers(conn, n: int, start: str, end: str, min_days: int = 200) -> list[str]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticker
        FROM stock_data
        WHERE date >= %s AND date <= %s
        GROUP BY ticker
        HAVING COUNT(*) >= %s
        ORDER BY COUNT(*) DESC, ticker
        LIMIT %s
    """, (start, end, min_days, n))
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tickers


def _load_close(conn, ticker: str, start: str, end: str) -> np.ndarray | None:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT close_price FROM stock_data
        WHERE ticker = %s AND date >= %s AND date <= %s
        ORDER BY date ASC
    """, (ticker, start, end))
    rows = cursor.fetchall()
    cursor.close()
    if not rows:
        return None
    close = np.array([r[0] for r in rows], dtype=float)
    for i in range(1, len(close)):
        if np.isnan(close[i]):
            close[i] = close[i - 1]
    close = close[~np.isnan(close)]
    return close if len(close) >= 100 else None


# ---------------------------------------------------------------------------
# Aggregate metrics helpers
# ---------------------------------------------------------------------------

def _agg(results: list[dict]) -> dict:
    """Mean of each metric across all per-stock results."""
    keys = ["MAE", "MAPE", "MSE", "RMSE", "R2"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


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
        f"{elapsed:>6.1f}s"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stocks", type=int, default=3000)
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--seq-len",  type=int, default=60)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--windows",  default="16,32,64")
    parser.add_argument("--device",   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--start",    default="2015-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    args = parser.parse_args()

    windows = tuple(int(w) for w in args.windows.split(","))

    print("=" * 75)
    print("  DeepFractal Benchmark — all models from Table 1")
    print(f"  Stocks: {args.n_stocks}  |  seq_len={args.seq_len}  |  epochs={args.epochs}")
    print(f"  Date range: {args.start} to {args.end}")
    print("=" * 75)

    conn = _get_conn()
    tickers = _load_tickers(conn, args.n_stocks, args.start, args.end)
    print(f"\nSelected {len(tickers)} tickers from stock_db\n")

    from baselines import (
        make_sequence_dataset,
        RNNModel, LSTMModel, GRUModel, ALSTMModel, VMDLSTMModel,
        train_baseline,
    )
    from fractal_features import extract_multiscale_features
    from model import DeepFractal, build_dataset, train, evaluate

    model_names = ["RNN", "LSTM", "GRU", "ALSTM", "VMD-LSTM", "DeepFractal"]
    all_results = {name: [] for name in model_names}
    timings     = {name: 0.0 for name in model_names}
    h = args.hidden
    n_done = 0

    for idx, ticker in enumerate(tickers):
        close = _load_close(conn, ticker, args.start, args.end)
        if close is None or len(close) < max(windows) + args.seq_len + 20:
            continue

        # --- sequence dataset (baselines) ---
        seq_ds = make_sequence_dataset(close, seq_len=args.seq_len)

        # --- fractal dataset (DeepFractal) ---
        try:
            scale_feats = extract_multiscale_features(close, windows=windows)
        except Exception:
            continue
        min_len = min(f.shape[0] for f in scale_feats.values())
        aligned = {w: scale_feats[w][-min_len:] for w in windows}
        end_idx = [len(close) - min_len + i for i in range(min_len)]
        targets = close[end_idx]
        frac_ds = build_dataset(aligned, targets)

        # --- train & eval all models ---
        specs = [
            ("RNN",      RNNModel(hidden=h),          False),
            ("LSTM",     LSTMModel(hidden=h),          False),
            ("GRU",      GRUModel(hidden=h),           False),
            ("ALSTM",    ALSTMModel(hidden=h),         False),
            ("VMD-LSTM", VMDLSTMModel(K=3, hidden=h), True),
        ]
        for name, model, is_vmd in specs:
            t0 = time.time()
            m = train_baseline(model, seq_ds, epochs=args.epochs, batch_size=args.batch,
                               lr=args.lr, patience=20, device=args.device,
                               seed=args.seed, is_vmd=is_vmd)
            timings[name] += time.time() - t0
            all_results[name].append(m)

        t0 = time.time()
        df_model = DeepFractal(fractal_dim=10, n_scales=len(windows),
                               feature_dim=64, tucker_rank=8,
                               latent_dim=16, dropout=0.3, use_vae=True)
        train(df_model, frac_ds, epochs=args.epochs, batch_size=args.batch,
              lr=args.lr, patience=20, use_fractal=True,
              device=args.device, seed=args.seed, verbose=False)
        m = evaluate(df_model, frac_ds, device=args.device)
        timings["DeepFractal"] += time.time() - t0
        all_results["DeepFractal"].append(m)

        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            agg = _agg(all_results["DeepFractal"])
            print(f"  [{n_done}/{len(tickers)}] {ticker}  "
                  f"DF RMSE={agg['RMSE']:.2f}  R²={agg['R2']:.4f}", flush=True)

    conn.close()

    if n_done == 0:
        print("No stocks processed — check DB connection and date range.")
        return

    # --- Aggregate across all stocks ---
    agg_results = {name: _agg(all_results[name]) for name in model_names}

    best = {
        "MSE":  min(m["MSE"]  for m in agg_results.values()),
        "RMSE": min(m["RMSE"] for m in agg_results.values()),
        "MAE":  min(m["MAE"]  for m in agg_results.values()),
        "MAPE": min(m["MAPE"] for m in agg_results.values()),
        "R2":   max(m["R2"]   for m in agg_results.values()),
    }

    print(f"\n{'='*75}")
    print(f"  Results averaged across {n_done} stocks")
    print(f"  {'Model':<12} {'MSE':>12}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>7}  {'R²':>7}  {'Time':>7}")
    print("  " + "-" * 71)
    for name in model_names:
        _print_row(name, agg_results[name], best, timings[name])
    print("=" * 75)
    print("  ✓ = best value in column")

    print("\n  Paper Table 1 reference (~3000 Chinese stocks, S&P 500 eval):")
    print(f"  {'Model':<12} {'MSE':>12}  {'RMSE':>9}  {'MAE':>8}  {'MAPE':>7}  {'R²':>7}")
    print("  " + "-" * 63)
    for name, mse, rmse, mae, mape, r2 in [
        ("RNN",         5362.06, 76.35, 58.94, 0.06, 0.99),
        ("LSTM",        4013.28, 68.37, 62.38, 0.05, 0.94),
        ("GRU",         3837.51, 62.76, 49.67, 0.05, 0.95),
        ("ALSTM",       3186.87, 58.39, 16.98, 0.07, 0.92),
        ("VMD-LSTM",     963.85, 43.68, 39.17, 0.03, 0.91),
        ("DeepFractal",  468.86, 19.63, 15.68, 0.02, 0.86),
    ]:
        marker = " ←" if name == "DeepFractal" else ""
        print(f"  {name:<12} {mse:>12,.2f}  {rmse:>9.2f}  {mae:>8.2f}  {mape:>7.2f}  {r2:>7.2f}{marker}")
    print()


if __name__ == "__main__":
    main()
