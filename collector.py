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
        ui.show_header("바이낸스 데이터 수집")

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
        df_binance = binance.collect_trades(
            symbol=symbol,
            start_time=start_time,
            save_format=save_format
        )

        if df_binance is not None:
            console.print("\n[bold]샘플 데이터:[/bold]")
            console.print(df_binance.head(10))

    # 업비트 수집
    if choice in ["2", "3"]:
        console.print("\n")
        ui.show_header("업비트 데이터 수집")

        market = ui.get_input("마켓 코드 입력 (예: KRW-BTC):").upper()
        max_trades = int(ui.get_input("최대 수집 개수 (예: 10000):") or "10000")

        upbit = UpbitTradeCollector()
        df_upbit = upbit.collect_trades(
            market=market,
            max_trades=max_trades,
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
