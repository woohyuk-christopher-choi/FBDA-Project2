#!/usr/bin/env python3
"""
바이낸스 API 클라이언트
"""

import requests
import time
from typing import Optional, Dict, List
from rich.console import Console

console = Console()


class BinanceAPIClient:
    """바이낸스 API 통신 클래스"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()

    def get_exchange_info(self, symbol: str) -> Optional[Dict]:
        """거래쌍 정보 조회"""
        try:
            response = self.session.get(f"{self.base_url}/exchangeInfo")
            data = response.json()

            for s in data['symbols']:
                if s['symbol'] == symbol:
                    return s

            return None
        except Exception as e:
            console.print(f"[red]거래쌍 정보 조회 실패: {e}[/red]")
            return None

    def fetch_trades(
            self,
            symbol: str,
            start_time: int,
            end_time: int,
            timeout: int = 30
    ) -> Optional[List[Dict]]:
        """
        거래 데이터 API 호출
        
        Args:
            symbol: 거래쌍
            start_time: 시작 시간 (밀리초)
            end_time: 종료 시간 (밀리초)
            timeout: 타임아웃 (초)
            
        Returns:
            거래 데이터 리스트 또는 None
        """
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            response = self.session.get(
                f"{self.base_url}/aggTrades",
                params=params,
                timeout=timeout
            )

            if response.status_code != 200:
                return None, response.status_code, response.text

            return response.json(), response.status_code, response.headers

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            raise e

    def fetch_klines(
            self,
            symbol: str,
            interval: str,
            start_time: int,
            end_time: int,
            timeout: int = 30
    ) -> Optional[List[Dict]]:
        """
        캠들 데이터 API 호출
        
        Args:
            symbol: 거래쌍
            interval: 캔들 간격 (1m, 5m, 15m, 30m, 1h, 1d 등)
            start_time: 시작 시간 (밀리초)
            end_time: 종료 시간 (밀리초)
            timeout: 타임아웃 (초)
            
        Returns:
            캠들 데이터 리스트 또는 None
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            response = self.session.get(
                f"{self.base_url}/klines",
                params=params,
                timeout=timeout
            )

            if response.status_code != 200:
                return None, response.status_code, response.text

            return response.json(), response.status_code, response.headers

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            raise e
