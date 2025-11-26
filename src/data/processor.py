#!/usr/bin/env python3
"""
데이터 처리 모듈 - DataFrame 변환 및 가공
"""

import polars as pl
import pyarrow as pa
from typing import List, Dict


class BinanceDataProcessor:
    """바이낸스 데이터 변환 클래스"""

    @staticmethod
    def convert_to_dataframe(trades: List[Dict]) -> pl.DataFrame:
        """거래 데이터를 Polars DataFrame으로 변환"""
        # Arrow 스키마 정의
        schema = pa.schema([
            ('agg_trade_id', pa.int64()),
            ('price', pa.float64()),
            ('quantity', pa.float64()),
            ('first_trade_id', pa.int64()),
            ('last_trade_id', pa.int64()),
            ('timestamp', pa.timestamp('ms')),
            ('is_buyer_maker', pa.bool_()),
            ('is_best_match', pa.bool_())
        ])

        # 데이터 변환
        data = {
            'agg_trade_id': [t['a'] for t in trades],
            'price': [float(t['p']) for t in trades],
            'quantity': [float(t['q']) for t in trades],
            'first_trade_id': [t['f'] for t in trades],
            'last_trade_id': [t['l'] for t in trades],
            'timestamp': [t['T'] for t in trades],
            'is_buyer_maker': [t['m'] for t in trades],
            'is_best_match': [t['M'] for t in trades]
        }

        # Polars DataFrame 생성
        df = pl.DataFrame(data)

        # timestamp는 정수로 유지, datetime 컬럼을 별도로 생성
        df = df.with_columns([
            pl.from_epoch(pl.col('timestamp'), time_unit='ms').alias('datetime')
        ])

        # 거래 금액 계산
        df = df.with_columns([
            (pl.col('price') * pl.col('quantity')).alias('amount')
        ])

        # 매수/매도 구분
        df = df.with_columns([
            pl.when(pl.col('is_buyer_maker'))
            .then(pl.lit('SELL'))
            .otherwise(pl.lit('BUY'))
            .alias('side')
        ])

        return df

    @staticmethod
    def convert_klines_to_dataframe(klines: List) -> pl.DataFrame:
        """캠들 데이터를 Polars DataFrame으로 변환"""
        # Binance klines 형식: [
        #   [
        #     1499040000000,      // Open time
        #     "0.01634790",       // Open
        #     "0.80000000",       // High
        #     "0.01575800",       // Low
        #     "0.01577100",       // Close
        #     "148976.11427815",  // Volume
        #     1499644799999,      // Close time
        #     "2434.19055334",    // Quote asset volume
        #     308,                // Number of trades
        #     "1756.87402397",    // Taker buy base asset volume
        #     "28.46694368",      // Taker buy quote asset volume
        #     "17928899.62484339" // Ignore
        #   ]
        # ]
        
        data = {
            'open_time': [k[0] for k in klines],
            'open': [float(k[1]) for k in klines],
            'high': [float(k[2]) for k in klines],
            'low': [float(k[3]) for k in klines],
            'close': [float(k[4]) for k in klines],
            'volume': [float(k[5]) for k in klines],
            'close_time': [k[6] for k in klines],
            'quote_volume': [float(k[7]) for k in klines],
            'trades': [k[8] for k in klines],
            'taker_buy_base': [float(k[9]) for k in klines],
            'taker_buy_quote': [float(k[10]) for k in klines]
        }
        
        df = pl.DataFrame(data)
        
        # timestamp 및 datetime 컬럼 추가
        df = df.with_columns([
            pl.col('open_time').alias('timestamp'),
            pl.from_epoch(pl.col('open_time'), time_unit='ms').alias('datetime')
        ])
        
        return df


class UpbitDataProcessor:
    """업비트 데이터 변환 클래스"""

    @staticmethod
    def convert_to_dataframe(trades: List[Dict]) -> pl.DataFrame:
        """거래 데이터를 Polars DataFrame으로 변환"""
        # Polars DataFrame 생성
        df = pl.DataFrame(trades)

        # 데이터 타입 변환
        df = df.with_columns([
            pl.col('trade_price').cast(pl.Float64),
            pl.col('trade_volume').cast(pl.Float64),
            pl.col('timestamp').cast(pl.Int64)
        ])

        # 타임스탬프를 datetime으로 변환
        df = df.with_columns([
            pl.from_epoch(pl.col('timestamp'), time_unit='ms').alias('datetime')
        ])

        # 거래 금액 계산
        df = df.with_columns([
            (pl.col('trade_price') * pl.col('trade_volume')).alias('amount')
        ])

        # 매수/매도 구분
        df = df.with_columns([
            pl.when(pl.col('ask_bid') == 'ASK')
            .then(pl.lit('SELL'))
            .otherwise(pl.lit('BUY'))
            .alias('side')
        ])

        # 시간 순서로 정렬
        df = df.sort('timestamp')

        return df

    @staticmethod
    def convert_candles_to_dataframe(candles: List[Dict]) -> pl.DataFrame:
        """캠들 데이터를 Polars DataFrame으로 변환"""
        # Upbit candles 형식:
        # {
        #   "market": "KRW-BTC",
        #   "candle_date_time_utc": "2023-01-01T00:00:00",
        #   "candle_date_time_kst": "2023-01-01T09:00:00",
        #   "opening_price": 19000000.0,
        #   "high_price": 19500000.0,
        #   "low_price": 18900000.0,
        #   "trade_price": 19200000.0,
        #   "timestamp": 1672531200000,
        #   "candle_acc_trade_price": 1234567890.0,
        #   "candle_acc_trade_volume": 123.456,
        #   "unit": 1  // 분 단위 (분봉의 경우)
        # }
        
        df = pl.DataFrame(candles)
        
        # 데이터 타입 변환
        df = df.with_columns([
            pl.col('opening_price').cast(pl.Float64).alias('open'),
            pl.col('high_price').cast(pl.Float64).alias('high'),
            pl.col('low_price').cast(pl.Float64).alias('low'),
            pl.col('trade_price').cast(pl.Float64).alias('close'),
            pl.col('candle_acc_trade_volume').cast(pl.Float64).alias('volume'),
            pl.col('candle_acc_trade_price').cast(pl.Float64).alias('quote_volume'),
            pl.col('timestamp').cast(pl.Int64)
        ])
        
        # datetime 컬럼 추가
        df = df.with_columns([
            pl.from_epoch(pl.col('timestamp'), time_unit='ms').alias('datetime')
        ])
        
        # 시간 순서로 정렬 (업비트는 역순으로 오므로 정렬 필요)
        df = df.sort('timestamp')
        
        return df


class DataMerger:
    """데이터 병합 및 중복 제거"""

    @staticmethod
    def merge_and_deduplicate(
            existing_df: pl.DataFrame,
            new_dfs: List[pl.DataFrame],
            id_column: str
    ) -> pl.DataFrame:
        """
        기존 데이터와 새 데이터를 병합하고 중복 제거
        
        Args:
            existing_df: 기존 DataFrame
            new_dfs: 새로운 DataFrame 리스트
            id_column: 중복 제거 기준 컬럼명
            
        Returns:
            병합된 DataFrame
        """
        all_dfs = [existing_df] + new_dfs
        merged_df = pl.concat(all_dfs)
        
        # 중복 제거 및 정렬
        merged_df = merged_df.unique(subset=[id_column], keep='last')
        merged_df = merged_df.sort('timestamp')
        
        return merged_df
