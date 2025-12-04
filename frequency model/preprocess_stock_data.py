#!/usr/bin/env python3
"""
S&P500 Stock Data Preprocessor

Converts raw Excel data to efficient Parquet format for fast loading.
Creates frequency-resampled files for 5min, 15min, 30min, 1h, 1d.

Usage:
    python preprocess_stock_data.py

Output:
    data/sp500/stocks_{freq}.parquet  - Stock prices by frequency
    data/sp500/market_{freq}.parquet  - Market index by frequency
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_excel(file_path: str):
    """Load raw S&P500 data from Excel file."""
    print(f"Loading Excel file: {file_path}")
    
    xl = pd.ExcelFile(file_path)
    
    # Get sheet names
    stock_sheets = [s for s in xl.sheet_names if s.startswith(('A', 'B', 'C'))]
    index_sheets = [s for s in xl.sheet_names if s.startswith('Index')]
    
    print(f"  Stock sheets: {stock_sheets}")
    print(f"  Index sheets: {index_sheets}")
    
    # Load all stock data
    all_stock_data = []
    
    for sheet_name in stock_sheets:
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        
        # Get asset names from row 0
        row0 = df.iloc[0].tolist()
        assets_str = None
        for val in row0:
            if pd.notna(val) and ';' in str(val):
                assets_str = str(val)
                break
        
        if assets_str:
            assets = [a.replace('.O', '') for a in assets_str.split(';')]
        else:
            assets = [f'Asset_{i}' for i in range(8)]
        
        print(f"  {sheet_name}: {len(assets)} assets - {assets[:3]}...")
        
        # Extract CLOSE prices for each asset
        for i, asset in enumerate(assets):
            col_timestamp = 1 + i * 6  # Timestamp column
            col_close = 1 + i * 6 + 4   # CLOSE column
            
            if col_close >= df.shape[1]:
                continue
            
            timestamps = pd.to_datetime(df.iloc[2:, col_timestamp], errors='coerce')
            prices = pd.to_numeric(df.iloc[2:, col_close], errors='coerce')
            
            asset_df = pd.DataFrame({
                'datetime': timestamps,
                'asset': asset,
                'close': prices
            }).dropna()
            
            if not asset_df.empty:
                all_stock_data.append(asset_df)
    
    # Combine all stock data
    combined = pd.concat(all_stock_data, ignore_index=True)
    
    # Pivot to wide format
    stock_prices = combined.pivot_table(
        index='datetime',
        columns='asset',
        values='close',
        aggfunc='first'
    ).sort_index()
    
    # Remove duplicate columns
    stock_prices = stock_prices.loc[:, ~stock_prices.columns.duplicated()]
    
    # Remove BRKa if both exist
    if 'BRKa' in stock_prices.columns and 'BRKb' in stock_prices.columns:
        stock_prices = stock_prices.drop(columns=['BRKa'])
    
    print(f"  Stock data shape: {stock_prices.shape}")
    
    # Load index data
    all_index_data = []
    for sheet_name in index_sheets:
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        
        timestamps = pd.to_datetime(df.iloc[2:, 1], errors='coerce')
        prices = pd.to_numeric(df.iloc[2:, 5], errors='coerce')
        
        index_df = pd.DataFrame({
            'datetime': timestamps,
            'close': prices
        }).dropna()
        
        all_index_data.append(index_df)
    
    if all_index_data:
        index_combined = pd.concat(all_index_data, ignore_index=True)
        index_combined = index_combined.drop_duplicates(subset='datetime')
        market_prices = index_combined.set_index('datetime')['close'].sort_index()
        market_prices.name = 'SP500'
    else:
        market_prices = stock_prices.mean(axis=1)
        market_prices.name = 'Market'
    
    print(f"  Market index shape: {len(market_prices)}")
    
    # Align indices
    common_idx = stock_prices.index.intersection(market_prices.index)
    stock_prices = stock_prices.loc[common_idx]
    market_prices = market_prices.loc[common_idx]
    
    # Remove stocks with too many NaNs (>50%)
    valid_cols = stock_prices.columns[stock_prices.notna().sum() > len(stock_prices) * 0.5]
    stock_prices = stock_prices[valid_cols]
    
    print(f"\nFinal data:")
    print(f"  Stocks: {len(stock_prices.columns)} assets")
    print(f"  Date range: {stock_prices.index[0]} to {stock_prices.index[-1]}")
    print(f"  Observations: {len(stock_prices):,}")
    
    return stock_prices, market_prices


def resample_and_save(stock_prices: pd.DataFrame, market_prices: pd.Series, output_dir: Path):
    """Resample to different frequencies and save as Parquet."""
    
    frequencies = {
        '1min': '1min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '1d': '1D'
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for freq_name, resample_rule in frequencies.items():
        print(f"\nProcessing {freq_name}...")
        
        # Resample using last price
        if freq_name == '1min':
            # 1min is base frequency, no resampling needed
            stocks_resampled = stock_prices.copy()
            market_resampled = market_prices.copy()
        else:
            stocks_resampled = stock_prices.resample(resample_rule).last().dropna(how='all')
            market_resampled = market_prices.resample(resample_rule).last().dropna()
        
        # Align
        common_idx = stocks_resampled.index.intersection(market_resampled.index)
        stocks_resampled = stocks_resampled.loc[common_idx]
        market_resampled = market_resampled.loc[common_idx]
        
        print(f"  Observations: {len(stocks_resampled):,}")
        
        # Save stocks
        stocks_path = output_dir / f"stocks_{freq_name}.parquet"
        stocks_resampled.to_parquet(stocks_path)
        print(f"  Saved: {stocks_path.name}")
        
        # Save market
        market_path = output_dir / f"market_{freq_name}.parquet"
        market_resampled.to_frame().to_parquet(market_path)
        print(f"  Saved: {market_path.name}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("S&P500 DATA PREPROCESSING")
    print("=" * 60)
    
    # Find data file
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    excel_files = list(data_dir.glob("S&P500*.xlsx"))
    if not excel_files:
        excel_files = list(data_dir.glob("*S&P500*.xlsx"))
    
    if not excel_files:
        print(f"ERROR: No S&P500 Excel file found in {data_dir}")
        return
    
    excel_file = excel_files[0]
    print(f"Using: {excel_file.name}")
    
    # Load raw data
    stock_prices, market_prices = load_raw_excel(str(excel_file))
    
    # Output directory
    output_dir = data_dir / "sp500"
    
    # Resample and save
    resample_and_save(stock_prices, market_prices, output_dir)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.parquet")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
