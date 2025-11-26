#!/usr/bin/env python3
"""
암호화폐 거래소 데이터 수집기 (Polars + Arrow)
바이낸스와 업비트의 거래 데이터를 수집합니다.
API 키 불필요 - 공개 데이터만 사용
"""

import time
from rich.console import Console
from rich.panel import Panel

from src.core import BinanceTradeCollector, UpbitTradeCollector
from src.utils import UIManager

console = Console()
ui = UIManager()


def main():
    """메인 실행 함수"""
    ui.show_header("암호화폐 거래소 데이터 수집기", "(Polars + Arrow)")

    console.print("\n[bold]사용 가능한 거래소:[/bold]")
    console.print("  [cyan]1.[/cyan] 바이낸스 (Binance)")
    console.print("  [cyan]2.[/cyan] 업비트 (Upbit)")
    console.print("  [cyan]3.[/cyan] 둘 다")
    console.print("  [red]0.[/red] 종료")

    choice = ui.get_input("\n선택 (0-3):").strip()

    if choice == "0":
        ui.show_error("프로그램을 종료합니다.")
        return

    # 데이터 타입 선택
    console.print("\n[bold]데이터 타입 선택:[/bold]")
    console.print("  [cyan]1.[/cyan] 체결 데이터 (Trades)")
    console.print("  [cyan]2.[/cyan] 1분봉 (1m)")
    console.print("  [cyan]3.[/cyan] 5분봉 (5m)")
    console.print("  [cyan]4.[/cyan] 15분봉 (15m)")
    console.print("  [cyan]5.[/cyan] 30분봉 (30m)")
    console.print("  [cyan]6.[/cyan] 1시간봉 (1h)")
    console.print("  [cyan]7.[/cyan] 일봉 (1d)")

    data_type_choice = ui.get_input("선택 (1-7, 기본 1):") or "1"

    data_type_map = {
        "1": ("trades", None),
        "2": ("1m", "1m"),
        "3": ("5m", "5m"),
        "4": ("15m", "15m"),
        "5": ("30m", "30m"),
        "6": ("1h", "1h"),
        "7": ("1d", "1d")
    }
    data_type, interval = data_type_map.get(data_type_choice, ("trades", None))

    # 저장 형식 선택
    console.print("\n[bold]저장 형식 선택:[/bold]")
    console.print("  [cyan]1.[/cyan] Arrow (최고 속도, Polars 최적화)")
    console.print("  [cyan]2.[/cyan] Parquet (최고 압축률)")
    console.print("  [cyan]3.[/cyan] CSV (범용성)")

    format_choice = ui.get_input("선택 (1-3, 기본 1):") or "1"

    save_format = {
        "1": "arrow",
        "2": "parquet",
        "3": "csv"
    }.get(format_choice, "arrow")

    # 바이낸스 수집
    if choice in ["1", "3"]:
        console.print("\n")
        ui.show_header(f"바이낸스 데이터 수집 ({data_type})")

        symbol = ui.get_input("거래쌍 입력 (예: BTCUSDT):").upper()

        console.print("\n[bold]수집 기간 설정:[/bold]")
        console.print("  [cyan]1.[/cyan] 최근 1시간")
        console.print("  [cyan]2.[/cyan] 최근 24시간")
        console.print("  [cyan]3.[/cyan] 최근 7일")
        console.print("  [cyan]4.[/cyan] 사용자 지정")

        period = ui.get_input("선택 (1-4):").strip()

        now = int(time.time() * 1000)

        if period == "1":
            start_time = now - 3600000  # 1시간
        elif period == "2":
            start_time = now - 86400000  # 24시간
        elif period == "3":
            start_time = now - 604800000  # 7일
        else:
            days = int(ui.get_input("과거 며칠 전부터?"))
            start_time = now - (days * 86400000)

        binance = BinanceTradeCollector()
        
        df_binance = binance.collect_data(
            symbol=symbol,
            data_type=data_type,
            interval=interval,
            start_time=start_time,
            save_format=save_format
        )

        if df_binance is not None:
            console.print("\n[bold]샘플 데이터:[/bold]")
            console.print(df_binance.head(10))

    # 업비트 수집
    if choice in ["2", "3"]:
        console.print("\n")
        ui.show_header(f"업비트 데이터 수집 ({data_type})")

        market = ui.get_input("마켓 코드 입력 (예: KRW-BTC):").upper()
        
        console.print("\n[bold]수집 기간 설정:[/bold]")
        console.print("  [cyan]1.[/cyan] 최근 1시간")
        console.print("  [cyan]2.[/cyan] 최근 24시간")
        console.print("  [cyan]3.[/cyan] 최근 7일")
        console.print("  [cyan]4.[/cyan] 사용자 지정")

        period = ui.get_input("선택 (1-4):").strip()

        # 캠들 간격에 따라 적절한 수집 개수 계산
        if data_type == "trades":
            # 거래 데이터는 개수 제한
            if period == "1":
                max_items = 3600  # 1시간
            elif period == "2":
                max_items = 10000  # 24시간
            elif period == "3":
                max_items = 50000  # 7일
            else:
                max_items = int(ui.get_input("최대 수집 개수:") or "10000")
        else:
            # 캠들 데이터는 시간 기반으로 개수 계산
            interval_minutes = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "30m": 30,
                "1h": 60,
                "1d": 1440
            }
            
            minutes = interval_minutes.get(data_type, 1)
            
            if period == "1":
                max_items = 60 // minutes  # 1시간
            elif period == "2":
                max_items = (24 * 60) // minutes  # 24시간
            elif period == "3":
                max_items = (7 * 24 * 60) // minutes  # 7일
            else:
                days = int(ui.get_input("과거 며칠 전부터?"))
                max_items = (days * 24 * 60) // minutes

        upbit = UpbitTradeCollector()
        
        df_upbit = upbit.collect_data(
            market=market,
            data_type=data_type,
            interval=interval,
            max_items=max_items,
            save_format=save_format
        )

        if df_upbit is not None:
            console.print("\n[bold]샘플 데이터:[/bold]")
            console.print(df_upbit.head(10))

    console.print("\n")
    ui.show_header(
        "✓ 모든 작업이 완료되었습니다!",
        f"데이터는 'data' 폴더에 {save_format.upper()} 형식으로 저장되었습니다."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        ui.show_warning("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        ui.show_error(f"예상치 못한 에러: {e}")
