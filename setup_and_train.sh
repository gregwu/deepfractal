#!/bin/bash
# Quick setup and training script for DeepFractal with S&P 500 tickers

set -e

echo "=================================================="
echo "  DeepFractal S&P 500 Training Setup"
echo "=================================================="

# 1. Install dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# 2. Initialize database
echo ""
echo "[2/3] Initializing PostgreSQL database..."
python init_db.py || echo "⚠ Database initialization skipped (PostgreSQL may not be available)"

# 3. Start training
echo ""
echo "[3/3] Starting training..."
echo ""
echo "Command: python demo.py --sp500 500"
echo "  - Trains on 500 S&P 500 tickers"
echo "  - Uses 5 years of data (2020-2024)"
echo "  - Results saved to PostgreSQL"
echo ""
echo "This will take several hours on CPU. For faster training, use GPU:"
echo "  python demo.py --sp500 500 --device cuda"
echo ""

read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python demo.py --sp500 500
else
    echo "Training skipped. Run manually when ready:"
    echo "  python demo.py --sp500 500"
fi

echo ""
echo "=================================================="
echo "  Training complete!"
echo "=================================================="
