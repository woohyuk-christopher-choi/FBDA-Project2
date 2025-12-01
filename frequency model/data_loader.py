"""
Data Loader for Low-Beta Anomaly Research

This module provides utilities to:
1. Load cryptocurrency data from Binance/Upbit (Arrow format)
2. Load market benchmark indices (Parquet format)
3. Prepare data for TARV beta estimation

Expected data structure:
- data/{exchange}/{symbol}/{exchange}_{symbol}_{freq}.arrow
- data/benchmark/{exchange}/benchmark_{freq}.parquet
"""

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "data"
    exchange: str = "binance"
    frequency: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CryptoDataLoader:
    """
    Load cryptocurrency data from Arrow files.
    
    Directory structure:
        data/{exchange}/{symbol}/{exchange}_{symbol}_{freq}.arrow
    
    Example:
        data/binance/BTCUSDT/binance_BTCUSDT_1m.arrow
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.base_path = Path(config.data_dir) / config.exchange
        
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from directory structure."""
        if not self.base_path.exists():
            print(f"⚠️  Data directory not found: {self.base_path}")
            return []
        
        symbols = []
        for item in self.base_path.iterdir():
            if item.is_dir():
                symbol = item.name
                expected_file = f"{self.config.exchange}_{symbol}_{self.config.frequency}.arrow"
                if (item / expected_file).exists():
                    symbols.append(symbol)
        
        return sorted(symbols)
    
    def load_symbol(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load data for a single symbol."""
        filename = f"{self.config.exchange}_{symbol}_{self.config.frequency}.arrow"
        filepath = self.base_path / symbol / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            return None
        
        try:
            # Read Arrow file
            with open(filepath, 'rb') as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
                df = table.to_pandas()
            
            # Set datetime index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            elif 'open_time' in df.columns:
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            # Filter by date range
            start = start_date or self.config.start_date
            end = end_date or self.config.end_date
            
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """Load close prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            df = self.load_symbol(symbol, start_date, end_date)
            if df is not None and price_column in df.columns:
                prices[symbol] = df[price_column]
        
        if not prices:
            return pd.DataFrame()
        
        prices_df = pd.DataFrame(prices)
        prices_df = prices_df.ffill(limit=5)  # Forward fill small gaps
        return prices_df


class MarketBenchmark:
    """Load market benchmark indices from Parquet files."""
    
    @staticmethod
    def load_benchmark_from_file(
        data_dir: str,
        exchange: str,
        frequency: str = "1m"
    ) -> Optional[pd.Series]:
        """
        Load pre-calculated benchmark.
        
        File structure:
            data/benchmark/{exchange}/benchmark_{freq}.parquet
            data/benchmark/upbit/upbit_benchmark_{freq}.parquet
        """
        base_path = Path(data_dir) / "benchmark" / exchange
        
        if exchange == 'upbit':
            filename = f"upbit_benchmark_{frequency}.parquet"
        else:
            filename = f"benchmark_{frequency}.parquet"
            
        filepath = base_path / filename
        
        if not filepath.exists():
            print(f"⚠️  Benchmark file not found: {filepath}")
            return None
            
        try:
            print(f"📉 Loading Market Benchmark: {filename}")
            df = pd.read_parquet(filepath)
            
            # Set datetime index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'timestamp' in df.columns:
                ts_dtype = df['timestamp'].dtype
                if 'datetime64' in str(ts_dtype):
                    df.set_index('timestamp', inplace=True)
                else:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)
            elif 'open_time' in df.columns:
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            # Select price column (priority: index_value > close > first float column)
            if 'index_value' in df.columns:
                series = df['index_value']
            elif 'close' in df.columns:
                series = df['close']
            else:
                float_cols = df.select_dtypes(include=['float', 'float64']).columns
                if len(float_cols) > 0:
                    series = df[float_cols[0]]
                else:
                    raise ValueError("No valid price column found in benchmark file")
                
            series.name = 'Market'
            return series
            
        except Exception as e:
            print(f"❌ Error loading benchmark {filepath}: {e}")
            return None


# Frequency to minutes per observation mapping
FREQUENCY_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}


def prepare_data_for_analysis(
    data_dir: str = "data",
    exchange: str = "binance",
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_observations: int = 10000,
    frequency: str = "1m"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare asset prices and market benchmark for analysis.
    
    Args:
        data_dir: Path to data directory
        exchange: Exchange name ('binance', 'upbit')
        symbols: List of symbols to load (None = all available)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_observations: Minimum observations required per asset
        frequency: Data frequency ('1m', '5m', '15m', '30m', '1h', '1d')
    
    Returns:
        Tuple of (assets_prices DataFrame, market_prices Series)
    """
    config = DataConfig(
        data_dir=data_dir,
        exchange=exchange,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date
    )
    
    loader = CryptoDataLoader(config)
    
    # 1. Get available symbols
    available = loader.get_available_symbols()
    if not available:
        raise ValueError(f"No data found in {data_dir}/{exchange}")
    
    print(f"📊 Found {len(available)} symbols in {exchange}")
    
    # 2. Filter symbols if specified
    if symbols:
        symbols_to_load = [s for s in symbols if s in available]
    else:
        symbols_to_load = available
    
    print(f"📥 Loading {len(symbols_to_load)} asset symbols...")
    
    # 3. Load asset prices
    prices = loader.load_multiple_symbols(
        symbols_to_load, 
        start_date, 
        end_date,
        price_column='close'
    )
    
    if prices.empty:
        raise ValueError("No asset data loaded")
    
    # 4. Filter by minimum observations
    valid_symbols = [col for col in prices.columns 
                     if prices[col].notna().sum() >= min_observations]
    prices = prices[valid_symbols]
    print(f"✅ Loaded assets: {len(valid_symbols)} symbols")

    # 5. Load market benchmark
    market_prices = MarketBenchmark.load_benchmark_from_file(
        data_dir=data_dir,
        exchange=exchange,
        frequency=frequency
    )
    
    if market_prices is None:
        raise ValueError(f"Could not load benchmark file for {exchange}")

    # 6. Align data (intersection of dates)
    common_index = prices.index.intersection(market_prices.index)
    
    if len(common_index) == 0:
        raise ValueError("No overlapping dates between assets and benchmark!")
        
    prices = prices.loc[common_index]
    market_prices = market_prices.loc[common_index]
    
    print(f"🔗 Aligned Data Range: {prices.index[0]} to {prices.index[-1]}")
    print(f"   Total Observations: {len(prices):,}")
    print(f"   Frequency: {frequency} ({FREQUENCY_MINUTES.get(frequency, 1)} min/obs)")
    
    return prices, market_prices
