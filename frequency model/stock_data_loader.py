"""
Stock Data Loader for S&P500 Analysis

Loads preprocessed stock data from Parquet files.
For raw Excel loading, use preprocess_stock_data.py first.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


# Frequency mapping for US stock market (390 min/day)
STOCK_FREQUENCY_MINUTES = {
    '1min': 1,
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1h': 60,
    '1d': 390
}


def load_preprocessed_data(
    data_dir: str,
    frequency: str = '5min'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load preprocessed S&P500 data from Parquet files.
    
    Args:
        data_dir: Path to data directory (contains sp500/ subfolder)
        frequency: Data frequency ('1min', '5min', '15min', '30min', '1h', '1d')
    
    Returns:
        Tuple of (stock_prices DataFrame, market_prices Series)
    """
    sp500_dir = Path(data_dir) / "sp500"
    
    if not sp500_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {sp500_dir}\n"
            "Run 'python preprocess_stock_data.py' first to generate data."
        )
    
    # Load stocks
    stocks_path = sp500_dir / f"stocks_{frequency}.parquet"
    if not stocks_path.exists():
        raise FileNotFoundError(f"Stock data not found: {stocks_path}")
    
    stock_prices = pd.read_parquet(stocks_path)
    print(f"Loaded stocks: {stock_prices.shape}")
    
    # Load market
    market_path = sp500_dir / f"market_{frequency}.parquet"
    if not market_path.exists():
        raise FileNotFoundError(f"Market data not found: {market_path}")
    
    market_df = pd.read_parquet(market_path)
    market_prices = market_df.iloc[:, 0]
    market_prices.name = 'SP500'
    print(f"Loaded market: {len(market_prices)} observations")
    
    # Align indices
    common_idx = stock_prices.index.intersection(market_prices.index)
    stock_prices = stock_prices.loc[common_idx]
    market_prices = market_prices.loc[common_idx]
    
    print(f"Date range: {stock_prices.index[0]} to {stock_prices.index[-1]}")
    
    return stock_prices, market_prices


def load_sp500_data(
    file_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load S&P500 constituent stocks and index data from Excel file.
    (Legacy function - use load_preprocessed_data for better performance)
    
    Args:
        file_path: Path to Excel file
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
    
    Returns:
        Tuple of (stock_prices DataFrame, market_prices Series)
    """
    print(f"Loading S&P500 data from: {file_path}")
    
    xl = pd.ExcelFile(file_path)
    
    # Sheet name patterns
    stock_sheets = [s for s in xl.sheet_names if s.startswith(('A', 'B', 'C'))]
    index_sheets = [s for s in xl.sheet_names if s.startswith('Index')]
    
    print(f"  Stock sheets: {stock_sheets}")
    print(f"  Index sheets: {index_sheets}")
    
    all_stock_data = []
    
    # Process each stock sheet group
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
            col_timestamp = 1 + i * 6
            col_close = 1 + i * 6 + 4
            
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
    
    if not all_stock_data:
        raise ValueError("No stock data loaded")
    
    combined = pd.concat(all_stock_data, ignore_index=True)
    
    stock_prices = combined.pivot_table(
        index='datetime',
        columns='asset',
        values='close',
        aggfunc='first'
    ).sort_index()
    
    stock_prices = stock_prices.loc[:, ~stock_prices.columns.duplicated()]
    
    print(f"  Combined stock data: {stock_prices.shape}")
    
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
        print(f"  S&P500 index data: {len(market_prices)} observations")
    else:
        print("  Warning: No index sheet found, creating equal-weighted market proxy")
        market_prices = stock_prices.mean(axis=1)
        market_prices.name = 'Market'
    
    # Align indices
    common_idx = stock_prices.index.intersection(market_prices.index)
    stock_prices = stock_prices.loc[common_idx]
    market_prices = market_prices.loc[common_idx]
    
    # Filter by date range
    if start_date:
        stock_prices = stock_prices[stock_prices.index >= start_date]
        market_prices = market_prices[market_prices.index >= start_date]
    if end_date:
        stock_prices = stock_prices[stock_prices.index <= end_date]
        market_prices = market_prices[market_prices.index <= end_date]
    
    # Remove assets with too many NaNs
    valid_cols = stock_prices.columns[stock_prices.notna().sum() > len(stock_prices) * 0.5]
    stock_prices = stock_prices[valid_cols]
    
    # Remove BRKa
    if 'BRKa' in stock_prices.columns and 'BRKb' in stock_prices.columns:
        stock_prices = stock_prices.drop(columns=['BRKa'])
        print("  Removed BRKa (keeping only BRKb)")
    
    print(f"\nFinal data:")
    print(f"   Stocks: {len(stock_prices.columns)} assets")
    print(f"   Date range: {stock_prices.index[0]} to {stock_prices.index[-1]}")
    print(f"   Observations: {len(stock_prices):,}")
    
    return stock_prices, market_prices


def resample_to_frequency(
    prices: pd.DataFrame,
    freq: str = '5min'
) -> pd.DataFrame:
    """
    Resample price data to a different frequency.
    
    Args:
        prices: DataFrame with datetime index and price columns
        freq: Target frequency ('5min', '15min', '30min', '1h', '1d')
    
    Returns:
        Resampled DataFrame (using last price in each period)
    """
    return prices.resample(freq).last().dropna(how='all')


if __name__ == "__main__":
    # Test loading
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Try preprocessed data first
    try:
        stocks, market = load_preprocessed_data(str(data_dir), '5min')
        print(f"\nLoaded preprocessed data successfully!")
        print(f"Stocks: {list(stocks.columns)[:5]}...")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nFalling back to Excel loading...")
        
        data_file = data_dir / "S&P500_index포함_분봉_20241126_20251126.xlsx"
        if data_file.exists():
            stocks, market = load_sp500_data(str(data_file))
        else:
            print(f"File not found: {data_file}")
