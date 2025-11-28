#!/usr/bin/env python3
"""
자동 데이터 수집 스크립트
Binance와 Upbit 벤치마크 생성에 필요한 모든 심볼과 캔들 타입을 순차적으로 수집합니다.
"""

import time
from pathlib import Path
from typing import List, Set
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src.core import BinanceTradeCollector, UpbitTradeCollector

console = Console()

# 벤치마크 생성에 필요한 모든 심볼 (create_benchmark.py에서 추출)
REBALANCING_PERIODS = [
    {"start": "2023-04-28", "end": "2023-05-26", "constituents": ["BTC", "ETH", "BNB", "XRP", "ADA", "MATIC", "SOL", "DOT", "LTC", "TRX"]},
    {"start": "2023-05-26", "end": "2023-06-30", "constituents": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "MATIC", "LTC", "TRX", "DOT"]},
    {"start": "2023-06-30", "end": "2023-07-28", "constituents": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "TRX", "LTC", "MATIC", "DOT"]},
    {"start": "2023-07-28", "end": "2023-09-01", "constituents": ["BTC", "ETH", "XRP", "BNB", "ADA", "SOL", "TRX", "MATIC", "LTC", "DOT"]},
    {"start": "2023-09-01", "end": "2023-09-29", "constituents": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "TRX", "DOT", "MATIC", "LTC"]},
    {"start": "2023-09-29", "end": "2023-10-27", "constituents": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "TRX", "DOT", "MATIC", "LTC"]},
    {"start": "2023-10-27", "end": "2023-12-01", "constituents": ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "TRX", "MATIC", "DOT", "LTC"]},
    {"start": "2023-12-01", "end": "2023-12-29", "constituents": ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "TRX", "LINK", "DOT", "LTC"]},
    {"start": "2023-12-29", "end": "2024-01-26", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "TRX", "LINK"]},
    {"start": "2024-01-26", "end": "2024-03-01", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "TRX", "DOT", "LINK"]},
    {"start": "2024-03-01", "end": "2024-03-29", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "TRX", "LINK", "DOT"]},
    {"start": "2024-03-29", "end": "2024-04-26", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "TRX"]},
    {"start": "2024-04-26", "end": "2024-05-31", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "TRX"]},
    {"start": "2024-05-31", "end": "2024-06-28", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT", "TRX"]},
    {"start": "2024-06-28", "end": "2024-07-26", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "TRX", "BCH", "DOT"]},
    {"start": "2024-07-26", "end": "2024-08-30", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "TRX", "AVAX", "BCH", "DOT"]},
    {"start": "2024-08-30", "end": "2024-09-27", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "TRX", "ADA", "AVAX", "BCH", "DOT"]},
    {"start": "2024-09-27", "end": "2024-11-01", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "TRX", "ADA", "AVAX", "BCH", "DOT"]},
    {"start": "2024-11-01", "end": "2024-11-29", "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "TRX", "ADA", "AVAX", "BCH", "DOT"]},
    {"start": "2024-11-29", "end": "2024-12-27", "constituents": ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "TRX", "AVAX", "BCH", "DOT"]},
    {"start": "2024-12-27", "end": "2025-01-31", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "AVAX", "DOT", "BCH"]},
    {"start": "2025-01-31", "end": "2025-02-28", "constituents": ["BTC", "ETH", "XRP", "SOL", "BNB", "ADA", "TRX", "LINK", "AVAX", "DOT"]},
    {"start": "2025-02-28", "end": "2025-03-28", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "AVAX", "LINK", "DOT"]},
    {"start": "2025-03-28", "end": "2025-05-02", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "AVAX", "LINK", "DOT"]},
    {"start": "2025-05-02", "end": "2025-05-30", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "DOT", "LINK", "AVAX"]},
    {"start": "2025-05-30", "end": "2025-06-27", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "DOT", "LINK", "AVAX"]},
    {"start": "2025-06-27", "end": "2025-08-01", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "AVAX", "BCH", "LINK"]},
    {"start": "2025-08-01", "end": "2025-08-29", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "BCH", "XLM", "LINK"]},
    {"start": "2025-08-29", "end": "2025-09-26", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "LINK", "HBAR", "XLM"]},
    {"start": "2025-09-26", "end": "2025-10-31", "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "BCH", "XLM", "LINK"]},
]

# 지원하는 캔들 타입
CANDLE_TYPES = ['1m', '5m', '15m', '30m', '1h', '1d']

# 업비트 벤치마크 구성 종목 (고정)
UPBIT_CONSTITUENTS = [
    "BTC",   # 비트코인
    "ETH",   # 이더리움
    "USDT",  # 테더
    "XRP",   # 리플
    "SOL",   # 솔라나
    "USDC",  # USD코인
    "TRX",   # 트론
    "DOGE",  # 도지코인
    "ADA",   # 에이다
    "BCH"    # 비트코인캐시
]

# 업비트 캔들 타입 매핑
UPBIT_CANDLE_MAP = {
    '1m': '1',
    '5m': '5',
    '15m': '15',
    '30m': '30',
    '1h': '60',
    '1d': 'day'
}

# 수집 기간 설정 (2년 = 730일)
COLLECTION_DAYS = 730


def get_all_unique_symbols() -> List[str]:
    """모든 고유한 심볼 추출"""
    symbols = set()
    for period in REBALANCING_PERIODS:
        symbols.update(period["constituents"])
    return sorted(list(symbols))


def check_existing_data(data_dir: Path, symbol: str, candle_type: str, exchange: str = 'binance') -> bool:
    """기존 데이터가 충분한지 확인"""
    if exchange == 'binance':
        possible_files = [
            data_dir / f"binance_{symbol}USDT_{candle_type}.parquet",
            data_dir / f"binance_{symbol}USDT_{candle_type}.arrow",
        ]
    else:  # upbit
        upbit_tf = UPBIT_CANDLE_MAP.get(candle_type, candle_type)
        possible_files = [
            data_dir / f"upbit_KRW-{symbol}_{upbit_tf}m.parquet",
            data_dir / f"upbit_KRW-{symbol}_{upbit_tf}m.arrow",
        ]
    
    for file_path in possible_files:
        if file_path.exists():
            # 파일이 존재하면 이미 수집된 것으로 간주
            return True
    
    return False


def display_collection_plan(symbols: List[str], candle_types: List[str], data_dir: Path, exchange: str = 'binance'):
    """수집 계획 표시"""
    exchange_name = "Binance" if exchange == 'binance' else "Upbit"
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]  {exchange_name} 자동 데이터 수집 계획  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"[bold]거래소:[/bold] {exchange_name}")
    console.print(f"[bold]수집 기간:[/bold] 최근 {COLLECTION_DAYS}일 (약 2년)")
    console.print(f"[bold]심볼 개수:[/bold] {len(symbols)}개")
    console.print(f"[bold]캔들 타입:[/bold] {', '.join(candle_types)}")
    console.print(f"[bold]총 작업 수:[/bold] {len(symbols) * len(candle_types)}개")
    console.print(f"[bold]저장 위치:[/bold] {data_dir}\n")
    
    # 기존 데이터 확인
    existing_count = 0
    missing_items = []
    
    for symbol in symbols:
        for candle_type in candle_types:
            if check_existing_data(data_dir, symbol, candle_type, exchange):
                existing_count += 1
            else:
                missing_items.append((symbol, candle_type))
    
    console.print(f"[bold green]기존 데이터:[/bold green] {existing_count}개")
    console.print(f"[bold yellow]수집 필요:[/bold yellow] {len(missing_items)}개\n")
    
    # 심볼 목록 표시
    console.print("[bold]수집 대상 심볼:[/bold]")
    for i, symbol in enumerate(symbols, 1):
        if exchange == 'binance':
            display_symbol = f"{symbol}USDT"
        else:
            display_symbol = f"KRW-{symbol}"
        console.print(f"  {i:2d}. {display_symbol}", end="")
        if i % 5 == 0:
            console.print()  # 5개마다 줄바꿈
        else:
            console.print("  ", end="")
    console.print("\n")
    
    return missing_items


def collect_all_data(
    symbols: List[str],
    candle_types: List[str],
    data_dir: Path,
    skip_existing: bool = True,
    save_format: str = "parquet",
    exchange: str = 'binance'
):
    """모든 심볼과 캔들 타입에 대해 데이터 수집"""
    
    # 수집 계획 표시
    missing_items = display_collection_plan(symbols, candle_types, data_dir, exchange)
    
    if not missing_items and skip_existing:
        console.print("[bold green]✓ 모든 데이터가 이미 수집되어 있습니다![/bold green]\n")
        return
    
    # 사용자 확인
    console.print("[bold yellow]위 계획대로 데이터 수집을 시작하시겠습니까?[/bold yellow]")
    console.print("  [cyan]1.[/cyan] 예 (기존 데이터 건너뛰기)")
    console.print("  [cyan]2.[/cyan] 예 (모든 데이터 재수집)")
    console.print("  [cyan]3.[/cyan] 아니오")
    
    choice = input("\n선택 (1-3): ").strip()
    
    if choice == "3":
        console.print("[yellow]수집을 취소했습니다.[/yellow]\n")
        return
    elif choice == "2":
        skip_existing = False
        items_to_collect = [(s, c) for s in symbols for c in candle_types]
    else:
        skip_existing = True
        items_to_collect = missing_items if missing_items else [(s, c) for s in symbols for c in candle_types]
    
    console.print(f"\n[bold green]데이터 수집을 시작합니다...[/bold green]\n")
    
    # Collector 초기화
    if exchange == 'binance':
        collector = BinanceTradeCollector(output_dir="file")
        # 시작/종료 시간 계산 (Binance는 밀리초)
        now = int(time.time() * 1000)
        start_time = now - (COLLECTION_DAYS * 86400000)  # 730일 전
    else:
        collector = UpbitTradeCollector(output_dir="file")
    
    # 통계
    total_items = len(items_to_collect)
    success_count = 0
    failed_items = []
    skipped_count = 0
    
    # 전체 진행 상황 표시
    start_timestamp = datetime.now()
    
    for idx, (symbol, candle_type) in enumerate(items_to_collect, 1):
        if exchange == 'binance':
            symbol_full = f"{symbol}USDT"
        else:
            symbol_full = f"KRW-{symbol}"
        
        # 기존 데이터 확인
        if skip_existing and check_existing_data(data_dir, symbol, candle_type, exchange):
            console.print(f"[dim]({idx}/{total_items}) {symbol_full} {candle_type}: 이미 존재함 (건너뜀)[/dim]")
            skipped_count += 1
            continue
        
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]진행 상황: {idx}/{total_items}[/bold]")
        console.print(f"[bold]심볼: {symbol_full} | 캔들: {candle_type}[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
        
        try:
            if exchange == 'binance':
                # Binance 데이터 수집
                df = collector.collect_klines(
                    symbol=symbol_full,
                    interval=candle_type,
                    start_time=start_time,
                    end_time=now,
                    max_klines=None,  # 제한 없음
                    save_format=save_format,
                    data_type=candle_type
                )
            else:
                # Upbit 데이터 수집
                # 캔들 타입별로 필요한 개수 계산
                interval_minutes = {
                    "1m": 1,
                    "5m": 5,
                    "15m": 15,
                    "30m": 30,
                    "1h": 60,
                    "1d": 1440
                }
                minutes = interval_minutes.get(candle_type, 1)
                max_items = (COLLECTION_DAYS * 24 * 60) // minutes
                
                upbit_interval = UPBIT_CANDLE_MAP.get(candle_type, candle_type)
                
                df = collector.collect_data(
                    market=symbol_full,
                    data_type=candle_type,
                    interval=upbit_interval,
                    max_items=max_items,
                    save_format=save_format
                )
            
            if df is not None and df.height > 0:
                success_count += 1
                console.print(f"[green]✓ 완료: {symbol_full} {candle_type} ({df.height:,}개 캔들)[/green]")
            else:
                failed_items.append((symbol_full, candle_type, "데이터 없음"))
                console.print(f"[yellow]⚠ 경고: {symbol_full} {candle_type} - 데이터 없음[/yellow]")
            
            # Rate limit 방지 - 요청 간 대기
            if idx < total_items:
                time.sleep(1)  # 1초 대기
                
        except KeyboardInterrupt:
            console.print(f"\n[bold red]사용자에 의해 중단되었습니다.[/bold red]")
            break
        except Exception as e:
            failed_items.append((symbol_full, candle_type, str(e)))
            console.print(f"[red]✗ 오류: {symbol_full} {candle_type} - {e}[/red]")
            continue
    
    # 최종 통계 표시
    end_timestamp = datetime.now()
    elapsed = end_timestamp - start_timestamp
    
    console.print("\n" + "="*70)
    console.print("[bold cyan]수집 완료![/bold cyan]")
    console.print("="*70 + "\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="green")
    
    table.add_row("총 작업 수", f"{total_items}개")
    table.add_row("성공", f"[green]{success_count}개[/green]")
    table.add_row("건너뜀", f"[dim]{skipped_count}개[/dim]")
    table.add_row("실패", f"[red]{len(failed_items)}개[/red]")
    table.add_row("소요 시간", str(elapsed).split('.')[0])
    table.add_row("평균 처리 시간", f"{elapsed.total_seconds() / max(total_items - skipped_count, 1):.1f}초/개")
    
    console.print(table)
    console.print()
    
    # 실패한 항목 표시
    if failed_items:
        console.print("[bold red]실패한 항목:[/bold red]")
        for symbol, candle_type, error in failed_items:
            console.print(f"  • {symbol} {candle_type}: {error}")
        console.print()
    
    # 다음 단계 안내
    if success_count > 0:
        console.print("[bold green]✓ 데이터 수집이 완료되었습니다![/bold green]")
        console.print("[bold]다음 단계:[/bold]")
        console.print("  1. 데이터 확인: python create_benchmark.py")
        console.print("  2. 벤치마크 생성: create_benchmark.py에서 'y' 선택\n")


def main():
    """메인 실행 함수"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  벤치마크 자동 데이터 수집기  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    # 거래소 선택
    console.print("[bold]거래소 선택:[/bold]")
    console.print("  [cyan]1.[/cyan] Binance (CMC Top 10 구성종목)")
    console.print("  [cyan]2.[/cyan] Upbit (고정 10개 종목)")
    console.print("  [cyan]3.[/cyan] 둘 다")
    
    exchange_choice = input("\n선택 (1-3, 기본 1): ").strip() or "1"
    
    # 데이터 디렉토리 설정
    data_dir = Path("file")
    data_dir.mkdir(exist_ok=True)
    
    # Binance 수집
    if exchange_choice in ["1", "3"]:
        console.print("\n[bold magenta]===== Binance 데이터 수집 =====[/bold magenta]")
        symbols = get_all_unique_symbols()
        process_exchange_collection(symbols, data_dir, 'binance')
    
    # Upbit 수집
    if exchange_choice in ["2", "3"]:
        console.print("\n[bold magenta]===== Upbit 데이터 수집 =====[/bold magenta]")
        symbols = UPBIT_CONSTITUENTS
        process_exchange_collection(symbols, data_dir, 'upbit')


def process_exchange_collection(symbols: List[str], data_dir: Path, exchange: str):
    """거래소별 데이터 수집 처리"""
    exchange_name = "Binance" if exchange == 'binance' else "Upbit"
    
    # 수집할 캔들 타입 선택
    console.print(f"\n[bold]{exchange_name} 수집할 캔들 타입 선택:[/bold]")
    console.print("  [cyan]1.[/cyan] 모든 캔들 타입 (1m, 5m, 15m, 30m, 1h, 1d)")
    console.print("  [cyan]2.[/cyan] 1분봉만 (1m)")
    console.print("  [cyan]3.[/cyan] 5분봉만 (5m)")
    console.print("  [cyan]4.[/cyan] 일봉만 (1d)")
    console.print("  [cyan]5.[/cyan] 사용자 지정")
    
    choice = input("\n선택 (1-5, 기본 1): ").strip() or "1"
    
    if choice == "1":
        selected_candles = CANDLE_TYPES
    elif choice == "2":
        selected_candles = ['1m']
    elif choice == "3":
        selected_candles = ['5m']
    elif choice == "4":
        selected_candles = ['1d']
    elif choice == "5":
        console.print("\n사용 가능한 캔들 타입:")
        for i, ct in enumerate(CANDLE_TYPES, 1):
            console.print(f"  {i}. {ct}")
        selections = input("\n번호를 공백으로 구분하여 입력 (예: 1 3 6): ").strip().split()
        selected_candles = [CANDLE_TYPES[int(s) - 1] for s in selections if s.isdigit() and 1 <= int(s) <= len(CANDLE_TYPES)]
    else:
        selected_candles = CANDLE_TYPES
    
    if not selected_candles:
        console.print("[red]유효한 캔들 타입이 선택되지 않았습니다.[/red]\n")
        return
    
    # 저장 형식 선택
    console.print("\n[bold]저장 형식 선택:[/bold]")
    console.print("  [cyan]1.[/cyan] Parquet (권장)")
    console.print("  [cyan]2.[/cyan] Arrow")
    
    format_choice = input("\n선택 (1-2, 기본 1): ").strip() or "1"
    save_format = "parquet" if format_choice == "1" else "arrow"
    
    # 데이터 수집 시작
    try:
        collect_all_data(
            symbols=symbols,
            candle_types=selected_candles,
            data_dir=data_dir,
            skip_existing=True,
            save_format=save_format,
            exchange=exchange
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]프로그램이 사용자에 의해 중단되었습니다.[/bold yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]예상치 못한 오류: {e}[/bold red]\n")


if __name__ == "__main__":
    main()
