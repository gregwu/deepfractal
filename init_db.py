#!/usr/bin/env python3
"""
Initialize the PostgreSQL database for DeepFractal results storage.
Run this once before training on S&P 500 tickers.

Usage:
    python init_db.py
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_database():
    """Create the deepfractal_results table if it doesn't exist."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "gangwu"),
            user=os.getenv("DB_USER", "gangwu"),
            password=os.getenv("DB_PASSWORD", "gangwu"),
        )
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deepfractal_results (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) UNIQUE NOT NULL,
                mae FLOAT,
                mape FLOAT,
                mse FLOAT,
                rmse FLOAT,
                r2 FLOAT,
                epochs INTEGER,
                batch_size INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create index on ticker for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker ON deepfractal_results(ticker);
        """)

        conn.commit()
        cursor.close()
        conn.close()

        print("✓ Database initialized successfully!")
        print("  Table: deepfractal_results")
        print("  Location: postgres://{}:{}@{}:{}/{}".format(
            os.getenv("DB_USER"),
            "****",
            os.getenv("DB_HOST"),
            os.getenv("DB_PORT"),
            os.getenv("DB_NAME"),
        ))
    except psycopg2.Error as e:
        print(f"✗ Database initialization failed: {e}")
        print("  Make sure PostgreSQL is running and .env credentials are correct")
        exit(1)

if __name__ == "__main__":
    init_database()
