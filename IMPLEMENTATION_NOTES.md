# DeepFractal Multi-Ticker Training — Implementation Summary

## What's Changed

Your DeepFractal project now supports training on 500 S&P 500 tickers with 5 years of data using database connectivity via .env configuration.

## Key Modifications

### 1. **Environment Variables (.env)**
- Database connection loads from `.env` (PostgreSQL credentials already in place)
- Supports: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

### 2. **Database Integration**
- Automatically connects to PostgreSQL (with graceful fallback if unavailable)
- Results saved to `deepfractal_results` table
- Each ticker's metrics (MAE, MAPE, MSE, RMSE, R²) stored with timestamp

### 3. **S&P 500 Support**
- `--sp500 N` flag trains on N tickers from S&P 500
- Auto-fetches ticker list from Wikipedia
- Falls back to hardcoded list if Wikipedia unavailable

### 4. **Data Range**
- Default: **2020-2024** (5 years, ~1,250 trading days per ticker)
- Customizable via `--data-start` and `--data-end` (YYYY-MM-DD format)
- Previous default was 2015-2024 (9 years); now uses 5-year window

### 5. **Multi-Ticker Training**
- Trains and evaluates each ticker independently
- Aggregated summary statistics across all tickers
- Failed tickers logged (insufficient data, fetch errors, etc.)
- Progress tracking: `[N/M] Training on TICKER`

## New Files

```
demo.py                  ← MODIFIED: Multi-ticker support, .env integration, DB save
requirements.txt         ← NEW: pip dependencies
init_db.py              ← NEW: Database initialization script
TRAINING_GUIDE.md       ← NEW: Comprehensive usage guide
setup_and_train.sh      ← NEW: Automated setup + training script
IMPLEMENTATION_NOTES.md ← NEW: This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database (Optional)
```bash
python init_db.py
```

### 3. Train on 500 S&P 500 Tickers
```bash
python demo.py --sp500 500
```

## Usage Examples

### Single Ticker (Default)
```bash
python demo.py                           # S&P 500 index (^GSPC)
python demo.py --ticker AAPL            # Apple
```

### Multiple S&P 500 Tickers
```bash
python demo.py --sp500 100              # 100 tickers
python demo.py --sp500 500              # 500 tickers (full S&P 500)
```

### Custom Date Range
```bash
python demo.py --sp500 500 --data-start 2018-01-01 --data-end 2024-12-31
```

### Performance Options
```bash
python demo.py --sp500 500 --device cuda --batch-size 128   # GPU training
python demo.py --sp500 10 --epochs 50                       # Quick test
```

## Command-Line Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `^GSPC` | Single ticker symbol |
| `--sp500` | 0 | Train on N S&P 500 tickers |
| `--windows` | `16,32,64` | Fractal feature windows |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--feature-dim` | 64 | Feature dimension |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--data-start` | `2020-01-01` | Data start date |
| `--data-end` | `2024-12-31` | Data end date |
| `--no-fractal-loss` | — | Use MSE only (no fractal loss) |
| `--no-vae` | — | Disable VAE bottleneck |

## Output Format

### Single Ticker
- 5-stage pipeline: Load → Extract → Build → Train → Evaluate
- Per-ticker metrics (MAE, MAPE, MSE, RMSE, R²)

### Multi-Ticker
- Progress for each ticker: `[N/M] Training on TICKER`
- Final summary with aggregated statistics:
  - Mean, Std, Min, Max for each metric across all tickers
  - Failed ticker count + sample list

## Database Schema

```sql
CREATE TABLE deepfractal_results (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    mae FLOAT,
    mape FLOAT,
    mse FLOAT,
    rmse FLOAT,
    r2 FLOAT,
    epochs INTEGER,
    batch_size INTEGER,
    timestamp TIMESTAMP,
    updated_at TIMESTAMP
);
```

Query results:
```sql
SELECT ticker, mae, rmse, r2 FROM deepfractal_results ORDER BY rmse ASC LIMIT 10;
```

## Estimated Training Time

| Setting | Time (CPU) | Time (GPU) |
|---------|-----------|-----------|
| 1 ticker, 200 epochs | ~1 min | ~10 sec |
| 100 tickers, 200 epochs | ~100 min | ~10 min |
| 500 tickers, 200 epochs | ~500 min (8.3 hrs) | ~50 min |

## Environment Setup

Your `.env` already contains:
```
DB_HOST=localhost
DB_PORT=5433
DB_NAME=gangwu
DB_USER=gangwu
DB_PASSWORD=gangwu
```

**Note**: Port is 5433 (not standard 5432). Make sure PostgreSQL is configured accordingly.

## Troubleshooting

### "Database connection failed"
- PostgreSQL not running? Start the service
- Wrong credentials? Update `.env`
- Training proceeds without database if unavailable

### "Insufficient data for TICKER"
- Some tickers have < 100 trading days in the 5-year window
- Logged as failed ticker; training continues

### "yfinance failed"
- Yahoo Finance API down or ticker invalid
- Fallback: Removed (previously used synthetic data)

### "Wikipedia fetch failed"
- Hardcoded S&P 500 ticker list used as fallback
- Reduces available tickers if Wikipedia unavailable

## Notes

- **Reproducibility**: Random seeds not set; results vary between runs
- **Data Quality**: Uses OHLC data from yfinance (free API)
- **Parallelization**: Sequential ticker processing (can be parallelized with multiprocessing)
- **Storage**: Results auto-saved to PostgreSQL if available
- **Monitoring**: No real-time monitoring; check database after training

## Paper Reference

Paper results (S&P 500 index):
```
Model       MSE      RMSE   MAE    MAPE   R²
RNN         5362.06  76.35  58.94  0.06   0.99
LSTM        4013.28  68.37  62.38  0.05   0.94
GRU         3837.51  62.76  49.67  0.05   0.95
ALSTM       3186.87  58.39  16.98  0.07   0.92
VMD-LSTM     963.85  43.68  39.17  0.03   0.91
Proposed     468.86  19.63  15.68  0.02   0.86  ← Your implementation
```

Your trained models will be compared against these benchmarks.
