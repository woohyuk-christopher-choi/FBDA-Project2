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
