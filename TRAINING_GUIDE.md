# DeepFractal Multi-Ticker Training Guide

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Database (Optional)
Edit `.env` with your PostgreSQL credentials:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gangwu
DB_USER=gangwu
DB_PASSWORD=gangwu
```

### 3. Initialize Database
```bash
python init_db.py
```

## Usage

### Train on Single Ticker (Default S&P 500 Index)
```bash
python demo.py
```

### Train on Single Custom Ticker
```bash
python demo.py --ticker AAPL
```

### Train on 500 S&P 500 Tickers (5 Years of Data)
```bash
python demo.py --sp500 500
```

### Train on 100 S&P 500 Tickers
```bash
python demo.py --sp500 100
```

### Custom Date Range (Default: 2020-2024)
```bash
python demo.py --sp500 500 --data-start 2019-01-01 --data-end 2024-12-31
```

### Customize Training Parameters
```bash
python demo.py --sp500 100 --epochs 100 --batch-size 32 --lr 0.0001 --feature-dim 64
```

### Disable Fractal Loss (MSE Only)
```bash
python demo.py --sp500 100 --no-fractal-loss
```

### Use GPU
```bash
python demo.py --sp500 100 --device cuda
```

## Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | `^GSPC` | Single ticker symbol |
| `--sp500` | 0 | Number of S&P 500 tickers to train on (0=single ticker) |
| `--windows` | `16,32,64` | Multi-scale fractal feature windows |
| `--epochs` | 200 | Number of training epochs |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 0.0001 | Learning rate |
| `--feature-dim` | 64 | Feature dimension |
| `--no-fractal-loss` | False | Disable fractal loss (use MSE only) |
| `--no-vae` | False | Disable VAE bottleneck |
| `--device` | cpu | Device: `cpu` or `cuda` |
| `--data-start` | 2020-01-01 | Data start date (YYYY-MM-DD) |
| `--data-end` | 2024-12-31 | Data end date (YYYY-MM-DD) |

## Output

### Single Ticker Output
```
==============================================================
  [1/1] Training on ^GSPC
==============================================================

[1/5] Loading price data...
  Loaded 1258 trading days from yfinance (^GSPC)

[2/5] Extracting fractal features at scales [16, 32, 64]...
  window= 16: 1243 samples × 10 features  (2.3s)
  window= 32: 1227 samples × 10 features  (2.3s)
  window= 64: 1195 samples × 10 features  (2.3s)

[3/5] Building train/val/test splits (60/20/20)...
  Train: 717  Val: 239  Test: 239

[4/5] Training DeepFractal model...
  Model parameters: 143,553
  Training time: 45.2s  Best val loss: 0.00234

[5/5] Evaluating on test set...
  Metric      Value
  ----
  MAE         15.6432
  MAPE         0.0198
  MSE         468.8600
  RMSE         19.6328
  R²           0.8600
```

### Multi-Ticker Summary
```
==============================================================
  SUMMARY: Trained on 500/500 tickers
==============================================================

  Aggregated Results (across 500 tickers):
  Metric      Mean        Std         Min         Max
  ----
  MAE         18.2341     12.3456      5.1234     89.3421
  RMSE        24.5678     18.9012      8.2341    123.4567
  R²           0.7234      0.2134      0.0123      0.9876
==============================================================
```

## Database Results

After training, results are automatically saved to PostgreSQL. Query them:

```sql
SELECT ticker, mae, rmse, r2, timestamp 
FROM deepfractal_results 
ORDER BY rmse ASC 
LIMIT 10;
```

## Example Workflows

### Quick Test (10 tickers, 50 epochs)
```bash
python demo.py --sp500 10 --epochs 50
```

### Full S&P 500 Training (500 tickers, 5 years)
```bash
python demo.py --sp500 500 --epochs 200
```

### Extended History (7 years)
```bash
python demo.py --sp500 500 --data-start 2018-01-01 --data-end 2024-12-31
```

### Performance Tuning
```bash
python demo.py --sp500 500 --device cuda --batch-size 128 --lr 0.0005
```

## Notes

- **Data**: Default uses 5 years (2020-2024). Adjust with `--data-start` and `--data-end`
- **Database**: Optional. If unavailable, training proceeds without storage
- **Tickers**: Auto-fetches S&P 500 list from Wikipedia; falls back to hardcoded list
- **Timing**: ~500 tickers × 200 epochs takes 24+ hours on CPU; use GPU for faster training
- **Results**: Saved to `deepfractal_results` table with timestamp
