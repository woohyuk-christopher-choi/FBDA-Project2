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

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate returns from prices.
        Strictly uses 'close' column if available, or float columns only.
        """
        # [수정] 1. 'close' 컬럼이 명시적으로 존재하면 그것만 사용 (단일 종목 OHLCV 데이터인 경우)
        if 'close' in prices.columns:
            target_prices = prices[['close']]
        else:
            # [수정] 2. 여러 종목인 경우: int(타임스탬프)를 제외하고 float(가격)만 선택
            target_prices = prices.select_dtypes(include=['float', 'float32', 'float64'])

        # 선택된 데이터가 비어있으면 원본에서 강제 변환 시도
        if target_prices.empty:
             target_prices = prices.apply(pd.to_numeric, errors='coerce')

        if method == 'log':
            # 로그 수익률: ln(Pt / Pt-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = np.log(target_prices / target_prices.shift(1))
        else:
            # 단순 수익률: (Pt - Pt-1) / Pt-1
            returns = target_prices.pct_change()
        
        # 무한대(inf)나 NaN 값 제거
        return returns.replace([np.inf, -np.inf], np.nan).dropna()

class MarketBenchmark:
    """
    Load or Create market benchmark indices.
    """
    
    @staticmethod
    def load_benchmark_from_file(
        data_dir: str,
        exchange: str,
        frequency: str = "1m"
    ) -> Optional[pd.Series]:
        """
        Load pre-calculated benchmark from parquet file.
        Structure: data/benchmark/{exchange}/[upbit_]benchmark_{freq}.parquet
        """
        base_path = Path(data_dir) / "benchmark" / exchange
        
        # 이미지 기반 파일명 규칙 적용
        if exchange == 'upbit':
            filename = f"upbit_benchmark_{frequency}.parquet"
        else:
            # binance 및 기본값
            filename = f"benchmark_{frequency}.parquet"
            
        filepath = base_path / filename
        
        if not filepath.exists():
            print(f"⚠️  Benchmark file not found: {filepath}")
            return None
            
        try:
            print(f"📉 Loading Market Benchmark: {filename}")
            # Parquet 파일 읽기
            df = pd.read_parquet(filepath)
            
            # 인덱스(날짜) 처리 (CryptoDataLoader와 동일한 로직)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            elif 'open_time' in df.columns:
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('datetime', inplace=True)
                
            # 'close' 가격을 벤치마크 지수로 사용 (없으면 첫 번째 컬럼 사용)
            if 'close' in df.columns:
                series = df['close']
            else:
                series = df.iloc[:, 0]
                
            series.name = 'Market'
            return series
            
        except Exception as e:
            print(f"❌ Error loading benchmark {filepath}: {e}")
            return None

    # (기존 계산 로직은 백업용으로 남겨두거나 삭제하셔도 됩니다)
    @staticmethod
    def create_equal_weighted_index(prices: pd.DataFrame, name: str = 'Market') -> pd.Series:
        normalized = prices / prices.iloc[0] * 100
        index = normalized.mean(axis=1)
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
    Prepare data loading benchmark from file.
    """
    config = DataConfig(
        data_dir=data_dir,
        exchange=exchange,
        frequency="1m",
        start_date=start_date,
        end_date=end_date
    )
    
    loader = CryptoDataLoader(config)
    
    # 1. Load Available Symbols
    available = loader.get_available_symbols()
    if not available:
        raise ValueError(f"No data found in {config.data_dir}/{exchange}")
    
    print(f"📊 Found {len(available)} symbols in {exchange}")
    
    if symbols:
        symbols_to_load = [s for s in symbols if s in available]
    else:
        symbols_to_load = available
    
    print(f"📥 Loading {len(symbols_to_load)} asset symbols...")
    
    # 2. Load Asset Prices
    prices = loader.load_multiple_symbols(
        symbols_to_load, 
        start_date, 
        end_date,
        price_column='close'
    )
    
    if prices.empty:
        raise ValueError("No asset data loaded")
        
    # Filter by minimum observations
    valid_symbols = [col for col in prices.columns if prices[col].notna().sum() >= min_observations]
    prices = prices[valid_symbols]
    print(f"✅ Loaded assets: {len(valid_symbols)} symbols")

    # 3. Load Market Benchmark from File (수정된 부분)
    market_prices = MarketBenchmark.load_benchmark_from_file(
        data_dir=data_dir,
        exchange=exchange,
        frequency="1m"
    )
    
    if market_prices is None:
        raise ValueError(f"Critical: Could not load benchmark file for {exchange}")

    # 4. Align Data (중요: 자산 데이터와 벤치마크 데이터의 날짜를 교집합으로 맞춤)
    # 벤치마크와 개별 자산의 기간이 다를 수 있으므로 공통된 기간만 남깁니다.
    common_index = prices.index.intersection(market_prices.index)
    
    if len(common_index) == 0:
        raise ValueError("No overlapping dates between Assets and Benchmark!")
        
    prices = prices.loc[common_index]
    market_prices = market_prices.loc[common_index]
    
    print(f"🔗 Aligned Data Range: {prices.index[0]} to {prices.index[-1]}")
    print(f"   Total Observations: {len(prices):,}")
    
    return prices, market_prices