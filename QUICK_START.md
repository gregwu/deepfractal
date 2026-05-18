# Quick Command Reference

## Setup (One-Time)
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database (optional, requires PostgreSQL)
python init_db.py
```

## Training Commands

### Train on 500 S&P 500 Tickers (5 Years: 2020-2024)
```bash
# Default: 200 epochs, CPU, full S&P 500
python demo.py --sp500 500

# Fast GPU version
python demo.py --sp500 500 --device cuda

# With custom data range
python demo.py --sp500 500 --data-start 2018-01-01 --data-end 2024-12-31

# Quick test: 10 tickers, 50 epochs
python demo.py --sp500 10 --epochs 50
```

### Train on Single Ticker
```bash
# S&P 500 index (default)
python demo.py

# Specific ticker
python demo.py --ticker AAPL

# S&P 500 with custom dates
python demo.py --data-start 2019-01-01 --data-end 2024-12-31
```

### Advanced Training
```bash
# GPU with higher batch size
python demo.py --sp500 500 --device cuda --batch-size 128

# Custom learning rate
python demo.py --sp500 100 --lr 0.0005

# MSE only (no fractal loss)
python demo.py --sp500 100 --no-fractal-loss

# Extended history (7 years)
python demo.py --sp500 500 --data-start 2018-01-01 --data-end 2024-12-31
```

## Monitor Results

### View in Database
```sql
-- Best performers
SELECT ticker, rmse, mae, r2 FROM deepfractal_results 
ORDER BY rmse ASC LIMIT 10;

-- Recent results
SELECT ticker, rmse, timestamp FROM deepfractal_results 
WHERE timestamp > NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- Statistics
SELECT 
    AVG(rmse) as avg_rmse,
    STDDEV(rmse) as std_rmse,
    MIN(rmse) as min_rmse,
    MAX(rmse) as max_rmse
FROM deepfractal_results;
```

## Full Example Workflow

```bash
# 1. Install and setup
pip install -r requirements.txt
python init_db.py

# 2. Quick test (10 tickers)
python demo.py --sp500 10 --epochs 50

# 3. Full training (500 tickers on GPU)
python demo.py --sp500 500 --device cuda --epochs 200

# 4. Check results
psql -U gangwu -d gangwu -c \
  "SELECT ticker, rmse, r2 FROM deepfractal_results ORDER BY rmse ASC LIMIT 20;"
```

## Key Parameters

- **--sp500 N**: Train on N S&P 500 tickers (default: 0 = single ticker)
- **--data-start YYYY-MM-DD**: Start date (default: 2020-01-01)
- **--data-end YYYY-MM-DD**: End date (default: 2024-12-31)
- **--epochs N**: Training epochs (default: 200)
- **--batch-size N**: Batch size (default: 64)
- **--device cpu|cuda**: Use CPU or GPU (default: cpu)
- **--lr FLOAT**: Learning rate (default: 0.0001)

## Expected Output Sample

```
============================================================
  [1/500] Training on AAPL
============================================================

[1/5] Loading price data...
  Loaded 1258 trading days from yfinance (AAPL)

[2/5] Extracting fractal features at scales [16, 32, 64]...
  window= 16: 1243 samples × 10 features  (2.1s)
  window= 32: 1227 samples × 10 features  (2.1s)
  window= 64: 1195 samples × 10 features  (2.1s)

[3/5] Building train/val/test splits (60/20/20)...
  Train: 717  Val: 239  Test: 239

[4/5] Training DeepFractal model...
  Model parameters: 143,553
  Training time: 42.3s  Best val loss: 0.00198

[5/5] Evaluating on test set...
  Metric      Value
  ----
  MAE         12.3456
  MAPE         0.0156
  MSE         382.4521
  RMSE         19.5533
  R²           0.8723
  Results saved to database for AAPL

...
```

## Timing Estimates

- **1 ticker, 200 epochs**: ~1 min (CPU) / ~10 sec (GPU)
- **10 tickers, 200 epochs**: ~10 min (CPU) / ~1 min (GPU)
- **100 tickers, 200 epochs**: ~100 min (CPU) / ~10 min (GPU)
- **500 tickers, 200 epochs**: ~8+ hours (CPU) / ~50 min (GPU)

Use `--epochs 50` for quick tests.
