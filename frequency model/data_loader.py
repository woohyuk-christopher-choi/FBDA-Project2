"""
Data Loader for Low-Beta Anomaly Research

This module provides utilities to:
1. Load cryptocurrency data from Binance/Upbit
2. Load equity data 
3. Create market benchmarks
4. Prepare data for TARV beta estimation

Expected data structure:
- data/binance/{symbol}_1m.arrow  (1-minute candles)
- data/upbit/{symbol}_1m.arrow
- data/equities/{symbol}_1m.arrow
"""

import os
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc  # [수정됨] PyArrow IPC 모듈 추가
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "data"
    exchange: str = "binance"  # 'binance', 'upbit', 'equities'
    frequency: str = "1m"       # '1m', '5m', '15m', '1h', '1d'
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class CryptoDataLoader:
    """
    Load and prepare cryptocurrency data for analysis.
    Handles nested directory structure: data/{exchange}/{symbol}/{exchange}_{symbol}_{freq}.arrow
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        # base_path는 data/binance 까지를 가리킴
        self.base_path = Path(config.data_dir) / config.exchange 
        
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols by checking subdirectories."""
        if not self.base_path.exists():
            print(f"⚠️  Data directory not found: {self.base_path}")
            return []
        
        symbols = []
        
        # data/binance/ 폴더 내부의 모든 항목을 순회
        for item in self.base_path.iterdir():
            # 항목이 디렉토리인지 확인 (예: ADAUSDT 폴더)
            if item.is_dir():
                symbol = item.name  # 폴더 이름이 곧 심볼 (예: ADAUSDT)
                
                # 해당 폴더 안에 우리가 원하는 1m 파일이 실제로 있는지 확인
                # 예상 파일명: binance_ADAUSDT_1m.arrow
                expected_filename = f"{self.config.exchange}_{symbol}_{self.config.frequency}.arrow"
                expected_filepath = item / expected_filename
                
                if expected_filepath.exists():
                    symbols.append(symbol)
        
        return sorted(symbols)
    
    def load_symbol(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol.
        """
        # 파일 경로 구성 수정:
        # data/binance/{symbol}/binance_{symbol}_1m.arrow
        filename = f"{self.config.exchange}_{symbol}_{self.config.frequency}.arrow"
        filepath = self.base_path / symbol / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            return None
        
        try:
            # PyArrow IPC 방식으로 파일 읽기
            with open(filepath, 'rb') as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
                df = table.to_pandas()
            
            # Ensure datetime index
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
    
    # ... 나머지 메서드(load_multiple_symbols, calculate_returns)는 그대로 유지 ...
    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        prices = {}
        for symbol in symbols:
            df = self.load_symbol(symbol, start_date, end_date)
            if df is not None and price_column in df.columns:
                prices[symbol] = df[price_column]
        if not prices:
            return pd.DataFrame()
        prices_df = pd.DataFrame(prices)
        prices_df = prices_df.ffill(limit=5)
        return prices_df

    def calculate_returns(self, prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        return returns.dropna()

class MarketBenchmark:
    """
    Create market benchmark indices.
    """
    
    @staticmethod
    def create_equal_weighted_index(
        prices: pd.DataFrame,
        name: str = 'Market'
    ) -> pd.Series:
        """
        Create equal-weighted market index from asset prices.
        
        Args:
            prices: DataFrame of asset prices
            name: Name for the index
        
        Returns:
            Series of market index prices
        """
        # Normalize all prices to start at 100
        normalized = prices / prices.iloc[0] * 100
        
        # Equal-weighted average
        index = normalized.mean(axis=1)
        index.name = name
        
        return index
    
    @staticmethod
    def create_market_cap_weighted_index(
        prices: pd.DataFrame,
        market_caps: Dict[str, float],
        name: str = 'Market'
    ) -> pd.Series:
        """
        Create market-cap weighted index.
        
        Args:
            prices: DataFrame of asset prices
            market_caps: Dict of symbol -> market cap
            name: Name for the index
        
        Returns:
            Series of market index prices
        """
        # Get weights
        total_cap = sum(market_caps.get(s, 0) for s in prices.columns)
        if total_cap == 0:
            return MarketBenchmark.create_equal_weighted_index(prices, name)
        
        weights = {s: market_caps.get(s, 0) / total_cap for s in prices.columns}
        
        # Normalize prices
        normalized = prices / prices.iloc[0] * 100
        
        # Weighted average
        index = sum(normalized[s] * w for s, w in weights.items() if s in normalized.columns)
        index.name = name
        
        return index


def prepare_data_for_analysis(
    data_dir: str = "data",
    exchange: str = "binance",
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_observations: int = 10000
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for Low-Beta Anomaly analysis.
     
    This is the main entry point for loading real data.
     
    Args:
        data_dir: Path to data directory
        exchange: 'binance', 'upbit', or 'equities'
        symbols: List of symbols (if None, load all available)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_observations: Minimum observations required per asset
     
    Returns:
        Tuple of (assets_prices DataFrame, market_prices Series)
     
    Example:
        >>> assets_prices, market_prices = prepare_data_for_analysis(
        ...     data_dir="data",
        ...     exchange="binance",
        ...     symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', ...],
        ...     start_date="2024-01-01",
        ...     end_date="2024-06-30"
        ... )
    """
    config = DataConfig(
        data_dir=data_dir,
        exchange=exchange,
        frequency="1m",
        start_date=start_date,
        end_date=end_date
    )
    
    loader = CryptoDataLoader(config)
    
    # Get available symbols
    available = loader.get_available_symbols()
    
    if not available:
        raise ValueError(f"No data found in {config.data_dir}/{exchange}")
    
    print(f"📊 Found {len(available)} symbols in {exchange}")
    
    # Use specified symbols or all available
    if symbols:
        symbols_to_load = [s for s in symbols if s in available]
        if len(symbols_to_load) < len(symbols):
            missing = set(symbols) - set(symbols_to_load)
            print(f"⚠️  Missing symbols: {missing}")
    else:
        symbols_to_load = available
    
    print(f"📥 Loading {len(symbols_to_load)} symbols...")
    
    # Load prices
    prices = loader.load_multiple_symbols(
        symbols_to_load, 
        start_date, 
        end_date,
        price_column='close'
    )
    
    if prices.empty:
        raise ValueError("No data loaded")
    
    # Filter by minimum observations
    valid_symbols = [col for col in prices.columns if prices[col].notna().sum() >= min_observations]
    prices = prices[valid_symbols]
    
    print(f"✅ {len(valid_symbols)} symbols with >= {min_observations} observations")
    print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"   Total observations: {len(prices):,}")
    
    # Create market benchmark (equal-weighted)
    market_prices = MarketBenchmark.create_equal_weighted_index(prices, 'Market')
    
    return prices, market_prices