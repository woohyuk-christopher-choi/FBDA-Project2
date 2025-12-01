#!/usr/bin/env python3
"""
Low-Beta Anomaly Analysis Entry Point
Runs TARV-based portfolio analysis on cryptocurrency data.
"""

from pathlib import Path
from portfolio import run_real_data_analysis


def main():
    """Run the analysis."""
    # Get data directory relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Analysis parameters
    config = {
        'data_dir': str(data_dir),       # Absolute path to data
        'exchange': 'binance',           # Exchange to analyze
        'start_date': '2023-11-27',      # Start date
        'end_date': '2025-11-26',        # End date
        'use_tarv': True,                # Use TARV for beta estimation
        'n_portfolios': 3,               # Number of portfolios (quartiles)
        'rebalance_frequency': 'weekly', # Weekly rebalancing (optimal for 1m data)
        'window_days': 30,               # Rolling window for beta
        'min_observations': 10000,       # Minimum observations per asset
    }
    
    print(f"\n{'='*60}")
    print(f"Low-Beta Anomaly Analysis: {config['exchange'].upper()}")
    print(f"Period: {config['start_date']} to {config['end_date']}")
    print(f"{'='*60}\n")
    
    try:
        run_real_data_analysis(**config)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()