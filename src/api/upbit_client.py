#!/usr/bin/env python3
"""
업비트 API 클라이언트
"""

import requests
from typing import Optional, Dict, List
from rich.console import Console

console = Console()


class UpbitAPIClient:
    """업비트 API 통신 클래스"""

    def __init__(self):
        self.base_url = "https://api.upbit.com/v1"
        self.session = requests.Session()

    def get_market_info(self, market: str) -> Optional[Dict]:
        """마켓 정보 조회"""
        try:
            response = self.session.get(f"{self.base_url}/market/all")
            markets = response.json()

            for m in markets:
                if m['market'] == market:
                    return m

            return None
        except Exception as e:
            console.print(f"[red]마켓 정보 조회 실패: {e}[/red]")
            return None

    def fetch_trades(
            self,
            market: str,
            count: int = 500,
            cursor: Optional[str] = None,
            timeout: int = 30
    ) -> Optional[List[Dict]]:
        """
        거래 데이터 API 호출
        
        Args:
            market: 마켓 코드
            count: 조회 개수
            cursor: 페이지네이션 커서
            timeout: 타임아웃 (초)
            
        Returns:
            거래 데이터 리스트 또는 None
        """
        params = {
            "market": market,
            "count": count
        }

        if cursor:
            params["cursor"] = cursor

        try:
            response = self.session.get(
                f"{self.base_url}/trades/ticks",
                params=params,
                timeout=timeout
            )

            if response.status_code != 200:
                return None, response.status_code, response.text

            return response.json(), response.status_code, None

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            raise e

    def fetch_candles(
            self,
            market: str,
            interval: str,
            count: int = 200,
            to: Optional[str] = None,
            timeout: int = 30
    ) -> Optional[List[Dict]]:
        """
        캠들 데이터 API 호출
        
        Args:
            market: 마켓 코드
            interval: 캠들 간격 (1, 5, 15, 30, 60, 240, day, week, month)
            count: 조회 개수 (최대 200)
            to: 마지막 캠들 시각 (yyyy-MM-dd'T'HH:mm:ss'Z' or yyyy-MM-dd HH:mm:ss)
            timeout: 타임아웃 (초)
            
        Returns:
            캠들 데이터 리스트 또는 None
        """
        # interval에 따라 endpoint 결정
        if interval in ['1', '5', '15', '30', '60', '240']:
            endpoint = f"{self.base_url}/candles/minutes/{interval}"
        elif interval == 'day':
            endpoint = f"{self.base_url}/candles/days"
        elif interval == 'week':
            endpoint = f"{self.base_url}/candles/weeks"
        elif interval == 'month':
            endpoint = f"{self.base_url}/candles/months"
        else:
            raise ValueError(f"지원하지 않는 interval: {interval}")

        params = {
            "market": market,
            "count": min(count, 200)
        }

        if to:
            params["to"] = to

        try:
            response = self.session.get(
                endpoint,
                params=params,
                timeout=timeout
            )

            if response.status_code != 200:
                return None, response.status_code, response.text

            return response.json(), response.status_code, None

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            raise e
