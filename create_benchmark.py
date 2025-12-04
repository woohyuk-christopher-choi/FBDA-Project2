"""
Binance Equal-Weighted Benchmark Generator
Creates benchmark candles with monthly rebalancing and equal weights
Supports multiple timeframes: 1m, 5m, 15m, 30m, 1h, 1d
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# Supported timeframes
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '1d']
TIMEFRAME_INTERVALS = {
    '1m': ('1m', '1m'),
    '5m': ('5m', '5m'),
    '15m': ('15m', '15m'),
    '30m': ('30m', '30m'),
    '1h': ('1h', '1h'),
    '1d': ('1d', '1d')
}

# Rebalancing periods from list.md (filtered to last 2 years: 2023-12-01 onwards)
REBALANCING_PERIODS = [
    {
        "start": "2023-12-01",
        "end": "2023-12-29",
        "constituents": ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "TRX", "LINK", "DOT", "LTC"]
    },
    {
        "start": "2023-12-29",
        "end": "2024-01-26",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "TRX", "LINK"]
    },
    {
        "start": "2024-01-26",
        "end": "2024-03-01",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "TRX", "DOT", "LINK"]
    },
    {
        "start": "2024-03-01",
        "end": "2024-03-29",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "TRX", "LINK", "DOT"]
    },
    {
        "start": "2024-03-29",
        "end": "2024-04-26",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "TRX"]
    },
    {
        "start": "2024-04-26",
        "end": "2024-05-31",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "TRX"]
    },
    {
        "start": "2024-05-31",
        "end": "2024-06-28",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT", "TRX"]
    },
    {
        "start": "2024-06-28",
        "end": "2024-07-26",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "TRX", "BCH", "DOT"]
    },
    {
        "start": "2024-07-26",
        "end": "2024-08-30",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "TRX", "AVAX", "BCH", "DOT"]
    },
    {
        "start": "2024-08-30",
        "end": "2024-09-27",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "TRX", "ADA", "AVAX", "BCH", "DOT"]
    },
    {
        "start": "2024-09-27",
        "end": "2024-11-01",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "TRX", "ADA", "AVAX", "BCH", "DOT"]
    },
    {
        "start": "2024-11-01",
        "end": "2024-11-29",
        "constituents": ["BTC", "ETH", "BNB", "SOL", "XRP", "TRX", "ADA", "AVAX", "BCH", "DOT"]
    },
    {
        "start": "2024-11-29",
        "end": "2024-12-27",
        "constituents": ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "TRX", "AVAX", "BCH", "DOT"]
    },
    {
        "start": "2024-12-27",
        "end": "2025-01-31",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "AVAX", "DOT", "BCH"]
    },
    {
        "start": "2025-01-31",
        "end": "2025-02-28",
        "constituents": ["BTC", "ETH", "XRP", "SOL", "BNB", "ADA", "TRX", "LINK", "AVAX", "DOT"]
    },
    {
        "start": "2025-02-28",
        "end": "2025-03-28",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "AVAX", "LINK", "DOT"]
    },
    {
        "start": "2025-03-28",
        "end": "2025-05-02",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "AVAX", "LINK", "DOT"]
    },
    {
        "start": "2025-05-02",
        "end": "2025-05-30",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "DOT", "LINK", "AVAX"]
    },
    {
        "start": "2025-05-30",
        "end": "2025-06-27",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "TRX", "DOT", "LINK", "AVAX"]
    },
    {
        "start": "2025-06-27",
        "end": "2025-08-01",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "AVAX", "BCH", "LINK"]
    },
    {
        "start": "2025-08-01",
        "end": "2025-08-29",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "BCH", "XLM", "LINK"]
    },
    {
        "start": "2025-08-29",
        "end": "2025-09-26",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "LINK", "HBAR", "XLM"]
    },
    {
        "start": "2025-09-26",
        "end": "2025-10-31",
        "constituents": ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "ADA", "BCH", "XLM", "LINK"]
    },
]

# Upbit Index constituents (fixed - no rebalancing)
# Based on Upbit's major trading pairs
UPBIT_CONSTITUENTS = [
    "BTC",  # 비트코인
    "ETH",  # 이더리움
    "USDT",  # 테더
    "XRP",  # 리플
    "SOL",  # 솔라나
    "USDC",  # USD코인
    "TRX",  # 트론
    "DOGE",  # 도지코인
    "ADA",  # 에이다
    "BCH"  # 비트코인캐시
]

# Calculate Upbit base date as 720 days ago from today
UPBIT_LOOKBACK_DAYS = 720
UPBIT_BASE_DATE = (datetime.now() - timedelta(days=UPBIT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
UPBIT_BASE_VALUE = 1000  # 업비트 지수 기준 값


def get_all_unique_tickers() -> List[str]:
    """Extract all unique ticker symbols from rebalancing periods"""
    tickers = set()
    for period in REBALANCING_PERIODS:
        tickers.update(period["constituents"])
    return sorted(list(tickers))


def check_data_availability(data_dir: Path, timeframes: Optional[List[str]] = None) -> Dict[str, Dict[str, bool]]:
    """
    Check which tickers have price data available for specified timeframes

    Args:
        data_dir: Directory containing price data
        timeframes: List of timeframes to check (e.g., ['1m', '5m', '1d'])
                   If None, checks all timeframes

    Returns:
        Dictionary mapping ticker -> timeframe -> availability (True/False)
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    availability = {}
    tickers = get_all_unique_tickers()

    console.print(
        f"\n[bold]Checking data availability for {len(tickers)} tickers across {len(timeframes)} timeframes...[/bold]\n")

    # Create table for better visualization
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="cyan")
    for tf in timeframes:
        table.add_column(tf, justify="center")

    for ticker in tickers:
        availability[ticker] = {}
        row = [f"{ticker}USDT"]

        for tf in timeframes:
            # Check for various possible file formats and naming conventions
            possible_files = [
                data_dir / f"binance_{ticker}USDT_{tf}.arrow",
                data_dir / f"binance_{ticker}USDT_{tf}.parquet",
                data_dir / f"{ticker}USDT_{tf}.arrow",
                data_dir / f"{ticker}USDT_{tf}.parquet",
                data_dir / "binance" / f"binance_{ticker}USDT_{tf}.arrow",
                data_dir / "binance" / f"binance_{ticker}USDT_{tf}.parquet",
                data_dir / "binance" / f"{ticker}USDT_{tf}.arrow",
                data_dir / "binance" / f"{ticker}USDT_{tf}.parquet",
                data_dir / "binance" / f"{ticker}USDT_{tf}.csv",
                data_dir / f"{ticker}USDT_{tf}.csv",
            ]

            has_data = any(f.exists() for f in possible_files)
            availability[ticker][tf] = has_data

            status = "✓" if has_data else "✗"
            color = "green" if has_data else "red"
            row.append(f"[{color}]{status}[/{color}]")

        table.add_row(*row)

    console.print(table)

    return availability


def generate_benchmark_summary():
    """Generate summary of what's needed for benchmark creation"""
    tickers = get_all_unique_tickers()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Binance Equal-Weighted Benchmark - Data Requirements  [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

    console.print(f"[bold]Rebalancing Periods:[/bold] {len(REBALANCING_PERIODS)}")
    console.print(f"[bold]Date Range:[/bold] {REBALANCING_PERIODS[0]['start']} to {REBALANCING_PERIODS[-1]['end']}")
    console.print(f"[bold]Unique Tickers:[/bold] {len(tickers)}")
    console.print(f"[bold]Constituents per Period:[/bold] 10 (equal-weighted)")
    console.print(f"[bold]Timeframes:[/bold] {', '.join(TIMEFRAMES)}\n")

    console.print("[bold]Required Tickers:[/bold]")
    for i, ticker in enumerate(tickers, 1):
        console.print(f"  {i:2d}. {ticker}USDT")

    console.print(f"\n[bold yellow]Data Collection Needed:[/bold yellow]")
    console.print(f"  • Candle data for all {len(tickers)} tickers")
    console.print(f"  • Timeframes: {', '.join(TIMEFRAMES)}")
    console.print(f"  • Date range: 2023-04-28 to 2025-10-31")
    console.print(f"  • Recommended format: Parquet or Arrow")
    console.print(f"  • Use collector.py to gather this data from Binance\n")

    # Check if data directory exists
    data_dir = Path("file")
    if data_dir.exists():
        availability = check_data_availability(data_dir, TIMEFRAMES)

        # Calculate statistics
        console.print(f"\n[bold]Data Completeness by Timeframe:[/bold]")
        for tf in TIMEFRAMES:
            available_count = sum(1 for ticker_data in availability.values() if ticker_data.get(tf, False))
            missing_count = len(tickers) - available_count
            percentage = (available_count / len(tickers)) * 100

            color = "green" if percentage == 100 else "yellow" if percentage > 50 else "red"
            console.print(f"  {tf:4s}: [{color}]{available_count:2d}/{len(tickers)}[/{color}] ({percentage:.1f}%)")

        # Check for completely missing tickers (no data at all)
        missing_all = []
        for ticker, ticker_data in availability.items():
            if not any(ticker_data.values()):
                missing_all.append(ticker)

        if missing_all:
            console.print(f"\n[bold red]Tickers with NO data for ANY timeframe:[/bold red]")
            for ticker in missing_all:
                console.print(f"  • {ticker}USDT")

        # Check for partially missing data
        partial_missing = []
        for ticker, ticker_data in availability.items():
            if any(ticker_data.values()) and not all(ticker_data.values()):
                missing_tfs = [tf for tf, avail in ticker_data.items() if not avail]
                partial_missing.append((ticker, missing_tfs))

        if partial_missing:
            console.print(f"\n[bold yellow]Tickers with PARTIAL data (missing some timeframes):[/bold yellow]")
            for ticker, missing_tfs in partial_missing:
                console.print(f"  • {ticker}USDT: missing {', '.join(missing_tfs)}")

        if missing_all or partial_missing:
            console.print(f"\n[bold yellow]Action Required:[/bold yellow]")
            console.print(f"  Run collector.py to collect missing data")
            console.print(f"  Example collection process:")
            console.print(f"    poetry run python collector.py")
            console.print(f"    -> Select: Binance (1)")
            console.print(f"    -> Select: desired timeframe (2-6)")
            console.print(f"    -> Enter: ticker (e.g., BTCUSDT)")
            console.print(f"    -> Period: Custom (4) -> ~900 days")
        else:
            console.print(f"\n[bold green]✓ All required data is available![/bold green]")
            console.print(f"  You can now create benchmark candles for all timeframes.")
    else:
        console.print(f"\n[bold red]No data directory found![/bold red]")
        console.print(f"  Please create 'data' folder and collect price data first.")
        console.print(f"  Use collector.py to gather candle data for all tickers.\n")


def check_upbit_data_availability(data_dir: Path, timeframes: Optional[List[str]] = None) -> Dict[str, Dict[str, bool]]:
    """
    Check which Upbit tickers have price data available for specified timeframes

    Args:
        data_dir: Directory containing price data
        timeframes: List of timeframes to check (e.g., ['1m', '5m', '1d'])
                   If None, checks all timeframes

    Returns:
        Dictionary mapping ticker -> timeframe -> availability (True/False)
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    availability = {}

    console.print(
        f"\n[bold]Checking Upbit data availability for {len(UPBIT_CONSTITUENTS)} tickers across {len(timeframes)} timeframes...[/bold]\n")

    # Create table for better visualization
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="magenta")
    for tf in timeframes:
        table.add_column(tf, justify="center")

    for ticker in UPBIT_CONSTITUENTS:
        availability[ticker] = {}
        row = [f"KRW-{ticker}"]

        for tf in timeframes:
            file_path = find_upbit_data_file(data_dir, ticker, tf)
            has_data = file_path is not None
            availability[ticker][tf] = has_data

            status = "✓" if has_data else "✗"
            color = "green" if has_data else "red"
            row.append(f"[{color}]{status}[/{color}]")

        table.add_row(*row)

    console.print(table)

    return availability


def generate_upbit_benchmark_summary():
    """Generate summary of what's needed for Upbit benchmark creation"""

    console.print("\n[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  Upbit Equal-Weighted Benchmark - Data Requirements   [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]\n")

    console.print(f"[bold]Index Type:[/bold] Fixed constituents (no rebalancing)")
    console.print(f"[bold]Base Date:[/bold] {UPBIT_BASE_DATE}")
    console.print(f"[bold]Base Value:[/bold] {UPBIT_BASE_VALUE}")
    console.print(f"[bold]Total Constituents:[/bold] {len(UPBIT_CONSTITUENTS)} (equal-weighted)")
    console.print(f"[bold]Timeframes:[/bold] {', '.join(TIMEFRAMES)}\n")

    console.print("[bold]Required Tickers:[/bold]")
    for i, ticker in enumerate(UPBIT_CONSTITUENTS, 1):
        console.print(f"  {i:2d}. KRW-{ticker}")

    console.print(f"\n[bold yellow]Data Collection Needed:[/bold yellow]")
    console.print(f"  • Candle data for all {len(UPBIT_CONSTITUENTS)} KRW pairs")
    console.print(f"  • Timeframes: {', '.join(TIMEFRAMES)}")
    console.print(f"  • Date range: From {UPBIT_BASE_DATE} to present")
    console.print(f"  • Recommended format: Parquet or Arrow")
    console.print(f"  • Use collector.py to gather this data from Upbit\n")

    # Check if data directory exists
    data_dir = Path("file")
    if data_dir.exists():
        availability = check_upbit_data_availability(data_dir, TIMEFRAMES)

        # Calculate statistics
        console.print(f"\n[bold]Data Completeness by Timeframe:[/bold]")
        for tf in TIMEFRAMES:
            available_count = sum(1 for ticker_data in availability.values() if ticker_data.get(tf, False))
            percentage = (available_count / len(UPBIT_CONSTITUENTS)) * 100

            color = "green" if percentage == 100 else "yellow" if percentage > 50 else "red"
            console.print(f"  {tf:4s}: [{color}]{available_count:2d}/{len(UPBIT_CONSTITUENTS)}[/{color}] ({percentage:.1f}%)")

        # Check for completely missing tickers (no data at all)
        missing_all = []
        for ticker, ticker_data in availability.items():
            if not any(ticker_data.values()):
                missing_all.append(ticker)

        if missing_all:
            console.print(f"\n[bold red]Tickers with NO data for ANY timeframe:[/bold red]")
            for ticker in missing_all:
                console.print(f"  • KRW-{ticker}")

        # Check for partially missing data
        partial_missing = []
        for ticker, ticker_data in availability.items():
            if any(ticker_data.values()) and not all(ticker_data.values()):
                missing_tfs = [tf for tf, avail in ticker_data.items() if not avail]
                partial_missing.append((ticker, missing_tfs))

        if partial_missing:
            console.print(f"\n[bold yellow]Tickers with PARTIAL data (missing some timeframes):[/bold yellow]")
            for ticker, missing_tfs in partial_missing:
                console.print(f"  • KRW-{ticker}: missing {', '.join(missing_tfs)}")

        if missing_all or partial_missing:
            console.print(f"\n[bold yellow]Action Required:[/bold yellow]")
            console.print(f"  Run collector.py to collect missing Upbit data")
            console.print(f"  Example collection process:")
            console.print(f"    poetry run python collector.py")
            console.print(f"    -> Select: Upbit (2)")
            console.print(f"    -> Select: desired timeframe")
            console.print(f"    -> Enter: ticker (e.g., KRW-BTC)")
        else:
            console.print(f"\n[bold green]✓ All required Upbit data is available![/bold green]")
            console.print(f"  You can now create benchmark candles for all timeframes.")
    else:
        console.print(f"\n[bold red]No data directory found![/bold red]")
        console.print(f"  Please create 'file' folder and collect price data first.")
        console.print(f"  Use collector.py to gather candle data for all tickers.\n")


def find_data_file(data_dir: Path, ticker: str, timeframe: str) -> Optional[Path]:
    """
    Find the data file for a given ticker and timeframe

    Args:
        data_dir: Directory containing price data
        ticker: Ticker symbol (without USDT)
        timeframe: Timeframe (e.g., '1m', '5m', '1d')

    Returns:
        Path to the data file if found, None otherwise
    """
    possible_files = [
        data_dir / f"binance_{ticker}USDT_{timeframe}.arrow",
        data_dir / f"binance_{ticker}USDT_{timeframe}.parquet",
        data_dir / f"{ticker}USDT_{timeframe}.arrow",
        data_dir / f"{ticker}USDT_{timeframe}.parquet",
        data_dir / "binance" / f"binance_{ticker}USDT_{timeframe}.arrow",
        data_dir / "binance" / f"binance_{ticker}USDT_{timeframe}.parquet",
        data_dir / "binance" / f"{ticker}USDT_{timeframe}.arrow",
        data_dir / "binance" / f"{ticker}USDT_{timeframe}.parquet",
        data_dir / "binance" / f"{ticker}USDT_{timeframe}.csv",
        data_dir / f"{ticker}USDT_{timeframe}.csv",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    return None


def find_upbit_data_file(data_dir: Path, ticker: str, timeframe: str) -> Optional[Path]:
    """
    Find the Upbit data file for a given ticker and timeframe

    Args:
        data_dir: Directory containing price data
        ticker: Ticker symbol (e.g., 'BTC')
        timeframe: Timeframe (e.g., '1m', '5m', '1d')

    Returns:
        Path to the data file if found, None otherwise
    """
    possible_files = [
        # New pattern: upbit_KRW_{ticker}_{timeframe}.arrow
        data_dir / f"upbit_KRW_{ticker}_{timeframe}.arrow",
        data_dir / f"upbit_KRW_{ticker}_{timeframe}.parquet",
        data_dir / f"upbit_{ticker}_{timeframe}.arrow",
        data_dir / f"upbit_{ticker}_{timeframe}.parquet",
        # In upbit subdirectory
        data_dir / "upbit" / f"upbit_KRW_{ticker}_{timeframe}.arrow",
        data_dir / "upbit" / f"upbit_KRW_{ticker}_{timeframe}.parquet",
        data_dir / "upbit" / f"upbit_{ticker}_{timeframe}.arrow",
        data_dir / "upbit" / f"upbit_{ticker}_{timeframe}.parquet",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    return None


def create_benchmark_candles(data_dir: Path, timeframe: str, output_dir: Path):
    """
    Create equal-weighted benchmark candles with monthly rebalancing

    Formula: I_t = I_0 × (1/D_t) × Σ(w_i × S_i,t / S_i,t_r)
    Where:
        - I_t = Index level at time t
        - I_0 = Base level (1000)
        - D_t = Divisor for continuity
        - w_i = Weight (0.1 for equal-weighted)
        - S_i,t = Price at time t
        - S_i,t_r = Price at rebalancing time

    Args:
        data_dir: Directory containing price data
        timeframe: Timeframe for benchmark candles (e.g., '1m', '5m', '1d')
        output_dir: Directory to save benchmark candles
    """
    console.print(f"\n[bold]Creating benchmark candles for {timeframe}...[/bold]\n")

    # Step 1: Load all constituent data
    console.print(f"[cyan]Step 1:[/cyan] Loading price data for all constituents...")

    all_data = {}
    tickers = get_all_unique_tickers()

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
    ) as progress:
        task = progress.add_task(f"Loading {len(tickers)} tickers...", total=len(tickers))

        for ticker in tickers:
            file_path = find_data_file(data_dir, ticker, timeframe)

            if file_path is None:
                console.print(f"[red]✗[/red] Missing data for {ticker}USDT")
                return None

            # Load data based on file format
            if file_path.suffix == '.parquet':
                df = pl.read_parquet(file_path)
            elif file_path.suffix == '.arrow':
                df = pl.read_ipc(file_path)
            elif file_path.suffix == '.csv':
                df = pl.read_csv(file_path)
            else:
                console.print(f"[red]✗[/red] Unsupported format for {ticker}USDT")
                return None

            # Standardize column names
            if 'timestamp' not in df.columns:
                date_cols = ['open_time', 'date', 'time', 'datetime']
                for col in date_cols:
                    if col in df.columns:
                        df = df.rename({col: 'timestamp'})
                        break

            if 'close' not in df.columns:
                console.print(f"[red]✗[/red] No 'close' column for {ticker}USDT")
                return None

            # Convert timestamp to datetime if needed
            if df['timestamp'].dtype != pl.Datetime:
                try:
                    # Try parsing as string first
                    df = df.with_columns([
                        pl.col('timestamp').str.to_datetime().alias('timestamp')
                    ])
                except:
                    # If that fails, try from milliseconds
                    try:
                        df = df.with_columns([
                            pl.from_epoch('timestamp', time_unit='ms').alias('timestamp')
                        ])
                    except:
                        console.print(f"[red]✗[/red] Cannot convert timestamp for {ticker}USDT")
                        return None

            all_data[ticker] = df.sort('timestamp')
            progress.update(task, advance=1)

    console.print(f"[green]✓[/green] Loaded data for {len(all_data)} tickers\n")

    # Step 2: Process each rebalancing period
    console.print(f"[cyan]Step 2:[/cyan] Processing {len(REBALANCING_PERIODS)} rebalancing periods...\n")

    all_period_indices = []
    last_index_value = None

    for i, period in enumerate(REBALANCING_PERIODS, 1):
        start_date = period['start']
        end_date = period['end']
        constituents = period['constituents']

        console.print(f"  Period {i:2d}: {start_date} to {end_date} | {len(constituents)} constituents")

        # Convert dates to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Collect data for this period
        period_data = {}
        for ticker in constituents:
            if ticker not in all_data:
                console.print(f"    [yellow]⚠[/yellow] {ticker} not in loaded data")
                continue

            df = all_data[ticker]

            # Filter by date range
            filtered = df.filter(
                (pl.col('timestamp') >= start_dt) &
                (pl.col('timestamp') < end_dt)
            )

            if filtered.height == 0:
                console.print(f"    [yellow]⚠[/yellow] No data for {ticker} in this period")
                continue

            period_data[ticker] = filtered

        if len(period_data) == 0:
            console.print(f"    [red]✗[/red] No data available for this period")
            continue

        # Calculate equal-weighted index for this period
        # Show progress for periods with substantial data (>1000 timestamps)
        show_progress_for_period = len(period_data) > 0
        if show_progress_for_period:
            # Estimate timestamps by checking first constituent
            first_ticker_data = next(iter(period_data.values()))
            show_progress_for_period = first_ticker_data.height > 1000

        period_index = calculate_equal_weighted_index(period_data, last_index_value, show_progress=show_progress_for_period)

        if period_index is not None and period_index.height > 0:
            all_period_indices.append(period_index)
            # Save last value for next period's continuity
            last_index_value = period_index['index_value'][-1]
            console.print(f"    [green]✓[/green] Generated {period_index.height} candles")

    if not all_period_indices:
        console.print(f"\n[red]✗[/red] No benchmark data generated")
        return None

    # Step 3: Combine all periods
    console.print(f"\n[cyan]Step 3:[/cyan] Combining all periods...")

    benchmark_df = pl.concat(all_period_indices).sort('timestamp')

    # Recalculate returns for the entire series
    benchmark_df = benchmark_df.with_columns([
        (pl.col('index_value') / pl.col('index_value').shift(1) - 1).alias('index_return')
    ])

    console.print(f"[green]✓[/green] Total benchmark candles: {benchmark_df.height}\n")

    # Step 4: Save to file
    console.print(f"[cyan]Step 4:[/cyan] Saving benchmark data...")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"binance_benchmark_{timeframe}.parquet"

    benchmark_df.write_parquet(output_file, compression='zstd')

    console.print(f"[green]✓[/green] Saved to: {output_file}\n")

    # Display summary statistics
    display_benchmark_summary(benchmark_df, timeframe)

    return benchmark_df


def calculate_equal_weighted_index(period_data: Dict[str, pl.DataFrame],
                                   last_index_value: Optional[float] = None,
                                   show_progress: bool = False) -> pl.DataFrame:
    """
    Calculate equal-weighted index for a single rebalancing period (OPTIMIZED VERSION)

    Formula: Index = Base × Σ(w_i × S_i,t / S_i,t_r) where w_i = 0.1

    This implements the official Binance CMC Top 10 calculation:
    1. Each constituent has 10% weight (equal-weighted)
    2. Price ratios are calculated relative to rebalancing time (first timestamp)
    3. Index continues from previous period for smooth transitions

    Args:
        period_data: Dictionary of ticker -> DataFrame with 'timestamp' and 'close' columns
        last_index_value: Index value from previous period (for continuity)
        show_progress: Whether to show progress bar during calculation

    Returns:
        DataFrame with columns: timestamp, index_value, index_return
    """
    if not period_data:
        return None

    if show_progress:
        console.print(f"    [dim]Optimizing data structure for {len(period_data)} tickers...[/dim]")

    # Step 1: Collect all unique timestamps from all tickers
    all_timestamps = set()
    for ticker, df in period_data.items():
        all_timestamps.update(df['timestamp'].to_list())

    if not all_timestamps:
        console.print(f"    [yellow]⚠[/yellow] No timestamps found in data")
        return None

    all_timestamps = sorted(list(all_timestamps))

    # Create base timestamp DataFrame
    base_ts_df = pl.DataFrame({'timestamp': all_timestamps}).with_columns([
        pl.col('timestamp').cast(pl.Datetime('ms'))
    ])

    # Step 2: Join all ticker data into a single wide DataFrame (much faster than repeated filtering)
    combined_df = base_ts_df

    for ticker, df in period_data.items():
        ticker_data = df.select(['timestamp', 'close']).rename({'close': f'close_{ticker}'})
        combined_df = combined_df.join(ticker_data, on='timestamp', how='left', coalesce=True)

    # Forward-fill and backward-fill missing values for all tickers at once
    close_cols = [col for col in combined_df.columns if col.startswith('close_')]
    combined_df = combined_df.with_columns([
        pl.col(col).fill_null(strategy='forward').fill_null(strategy='backward')
        for col in close_cols
    ])

    # Step 3: Filter to timestamps with at least 80% data coverage
    min_tickers_required = max(1, int(len(period_data) * 0.8))

    # Count non-null values per row
    combined_df = combined_df.with_columns([
        pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int32) for col in close_cols]).alias('_ticker_count')
    ])

    # Filter to rows with sufficient coverage
    combined_df = combined_df.filter(pl.col('_ticker_count') >= min_tickers_required).drop('_ticker_count')

    if combined_df.height == 0:
        console.print(f"    [yellow]⚠[/yellow] No timestamps with sufficient data coverage")
        return None

    # Step 4: Get base prices (first row = rebalancing time)
    base_row = combined_df.row(0, named=True)
    base_prices = {ticker: base_row[f'close_{ticker}'] for ticker in period_data.keys()}

    # Check for missing base prices
    for ticker, base_price in base_prices.items():
        if base_price is None or base_price == 0:
            console.print(f"    [yellow]⚠[/yellow] Invalid base price for {ticker}")
            return None

    # Step 5: Calculate price ratios for all tickers at once (vectorized)
    ratio_cols = []
    for ticker in period_data.keys():
        col_name = f'close_{ticker}'
        ratio_col_name = f'ratio_{ticker}'
        combined_df = combined_df.with_columns([
            (pl.col(col_name) / base_prices[ticker]).alias(ratio_col_name)
        ])
        ratio_cols.append(ratio_col_name)

    # Step 6: Calculate equal-weighted average ratio (mean of all ratios)
    combined_df = combined_df.with_columns([
        pl.mean_horizontal([pl.col(col) for col in ratio_cols]).alias('avg_ratio')
    ])

    # Get first average ratio for normalization
    first_avg_ratio = combined_df['avg_ratio'][0]

    # Step 7: Calculate index values
    if last_index_value is not None:
        # Continue from previous period
        combined_df = combined_df.with_columns([
            (last_index_value * (pl.col('avg_ratio') / first_avg_ratio)).alias('index_value')
        ])
    else:
        # First period: start from base level 1000
        combined_df = combined_df.with_columns([
            (1000 * pl.col('avg_ratio')).alias('index_value')
        ])

    # Step 8: Calculate returns
    combined_df = combined_df.with_columns([
        (pl.col('index_value') / pl.col('index_value').shift(1) - 1).alias('index_return')
    ])

    # Step 9: Keep only the essential columns
    result_df = combined_df.select(['timestamp', 'index_value', 'index_return'])

    return result_df


def display_benchmark_summary(benchmark_df: pl.DataFrame, timeframe: str):
    """Display summary statistics for the benchmark"""

    table = Table(title=f"Benchmark Summary - {timeframe}")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Basic stats
    table.add_row("Total Candles", f"{benchmark_df.height:,}")
    table.add_row("Start Date", str(benchmark_df['timestamp'][0]))
    table.add_row("End Date", str(benchmark_df['timestamp'][-1]))

    # Index values
    first_value = benchmark_df['index_value'][0]
    last_value = benchmark_df['index_value'][-1]
    total_return = (last_value / first_value - 1) * 100

    table.add_row("Starting Index", f"{first_value:.2f}")
    table.add_row("Ending Index", f"{last_value:.2f}")
    table.add_row("Total Return", f"{total_return:+.2f}%")

    # Return statistics
    returns = benchmark_df['index_return'].drop_nulls()
    if len(returns) > 0:
        avg_return = returns.mean() * 100
        std_return = returns.std() * 100
        min_return = returns.min() * 100
        max_return = returns.max() * 100

        table.add_row("Avg Return", f"{avg_return:+.4f}%")
        table.add_row("Std Dev", f"{std_return:.4f}%")
        table.add_row("Min Return", f"{min_return:+.4f}%")
        table.add_row("Max Return", f"{max_return:+.4f}%")

    console.print(table)


def create_upbit_benchmark_candles(data_dir: Path, timeframe: str, output_dir: Path):
    """
    Create Upbit equal-weighted benchmark candles

    Upbit Index Features:
    - Base Date: 2017-10-01
    - Base Value: 1000
    - Fixed constituents (no rebalancing)
    - 10 major KRW trading pairs
    - Equal-weighted approach

    Args:
        data_dir: Directory containing price data
        timeframe: Timeframe for benchmark candles (e.g., '1m', '5m', '1d')
        output_dir: Directory to save benchmark candles
    """
    console.print(f"\n[bold]Creating Upbit benchmark candles for {timeframe}...[/bold]\n")

    # Step 1: Load all constituent data
    console.print(f"[cyan]Step 1:[/cyan] Loading Upbit price data for {len(UPBIT_CONSTITUENTS)} constituents...")

    all_data = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
    ) as progress:
        task = progress.add_task(f"Loading {len(UPBIT_CONSTITUENTS)} tickers...", total=len(UPBIT_CONSTITUENTS))

        for ticker in UPBIT_CONSTITUENTS:
            file_path = find_upbit_data_file(data_dir, ticker, timeframe)

            if file_path is None:
                console.print(f"[red]✗[/red] Missing data for KRW-{ticker}")
                return None

            # Load data based on file format
            if file_path.suffix == '.parquet':
                df = pl.read_parquet(file_path)
            elif file_path.suffix == '.arrow':
                df = pl.read_ipc(file_path)
            elif file_path.suffix == '.csv':
                df = pl.read_csv(file_path)
            else:
                console.print(f"[red]✗[/red] Unsupported format for KRW-{ticker}")
                return None

            # Standardize column names for Upbit
            if 'timestamp' not in df.columns:
                # Upbit typically uses 'candle_date_time_kst' or similar
                date_cols = ['candle_date_time_kst', 'candle_date_time_utc', 'date', 'time', 'datetime']
                for col in date_cols:
                    if col in df.columns:
                        df = df.rename({col: 'timestamp'})
                        break

            # Upbit uses 'trade_price' instead of 'close'
            if 'close' not in df.columns:
                if 'trade_price' in df.columns:
                    df = df.rename({'trade_price': 'close'})
                else:
                    console.print(f"[red]✗[/red] No price column for KRW-{ticker}")
                    return None

            # Convert timestamp to datetime if needed
            if df['timestamp'].dtype != pl.Datetime:
                try:
                    df = df.with_columns([
                        pl.col('timestamp').str.to_datetime().alias('timestamp')
                    ])
                except:
                    try:
                        df = df.with_columns([
                            pl.from_epoch('timestamp', time_unit='ms').alias('timestamp')
                        ])
                    except:
                        console.print(f"[red]✗[/red] Cannot convert timestamp for KRW-{ticker}")
                        return None

            all_data[ticker] = df.sort('timestamp')
            progress.update(task, advance=1)

    console.print(f"[green]✓[/green] Loaded data for {len(all_data)} tickers\n")

    # Step 2: Filter data from base date
    console.print(f"[cyan]Step 2:[/cyan] Filtering data from {UPBIT_BASE_DATE}...")

    base_dt = datetime.strptime(UPBIT_BASE_DATE, '%Y-%m-%d')

    filtered_data = {}
    for ticker, df in all_data.items():
        filtered = df.filter(pl.col('timestamp') >= base_dt)
        if filtered.height == 0:
            console.print(f"[yellow]⚠[/yellow] No data for KRW-{ticker} from {UPBIT_BASE_DATE}")
            continue
        filtered_data[ticker] = filtered

    if len(filtered_data) == 0:
        console.print(f"[red]✗[/red] No data available from base date")
        return None

    console.print(f"[green]✓[/green] Filtered {len(filtered_data)} tickers\n")

    # Step 3: Calculate equal-weighted index
    console.print(f"[cyan]Step 3:[/cyan] Calculating equal-weighted index...")

    index_df = calculate_equal_weighted_index(filtered_data, None, show_progress=True)

    if index_df is None or index_df.height == 0:
        console.print(f"[red]✗[/red] Failed to calculate index")
        return None

    console.print(f"[green]✓[/green] Generated {index_df.height} candles\n")

    # Step 4: Save to file
    console.print(f"[cyan]Step 4:[/cyan] Saving Upbit benchmark data...")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"upbit_benchmark_{timeframe}.parquet"

    index_df.write_parquet(output_file, compression='zstd')

    console.print(f"[green]✓[/green] Saved to: {output_file}\n")

    # Display summary statistics
    display_benchmark_summary(index_df, f"Upbit {timeframe}")

    return index_df


if __name__ == "__main__":
    # First, ask which exchange summary to view
    console.print("\n[bold]Select which exchange data to check:[/bold]")
    console.print("  [cyan]1.[/cyan] Binance (CMC Top 10 with monthly rebalancing)")
    console.print("  [cyan]2.[/cyan] Upbit (Fixed 10 constituents)")
    console.print("  [cyan]3.[/cyan] Both")

    summary_choice = input("\nChoice (1-3): ").strip()

    if summary_choice in ['1', '3']:
        generate_benchmark_summary()

    if summary_choice in ['2', '3']:
        generate_upbit_benchmark_summary()

    # Optionally create benchmarks if data is available
    data_dir = Path("file")
    if data_dir.exists():
        console.print("\n[bold]Would you like to create benchmark candles? (y/n):[/bold] ", end="")
        response = input().strip().lower()

        if response == 'y':
            # Ask which exchange
            console.print("\n[bold]Select exchange:[/bold]")
            console.print("  [cyan]1.[/cyan] Binance (CMC Top 10 with monthly rebalancing)")
            console.print("  [cyan]2.[/cyan] Upbit (Fixed 10 constituents)")
            console.print("  [cyan]3.[/cyan] Both")

            exchange_choice = input("\nChoice (1-3): ").strip()

            # Ask which timeframes to generate
            console.print("\n[bold]Select timeframes to generate:[/bold]")
            console.print("  [cyan]1.[/cyan] All timeframes (1m, 5m, 15m, 30m, 1h, 1d)")
            console.print("  [cyan]2.[/cyan] Select specific timeframes")

            choice = input("\nChoice (1-2): ").strip()

            if choice == '1':
                selected_timeframes = TIMEFRAMES
            elif choice == '2':
                console.print("\n[bold]Available timeframes:[/bold]")
                for i, tf in enumerate(TIMEFRAMES, 1):
                    console.print(f"  {i}. {tf}")

                selections = input("\nEnter numbers separated by spaces (e.g., 1 3 6): ").strip().split()
                selected_timeframes = [TIMEFRAMES[int(s) - 1] for s in selections if
                                       s.isdigit() and 1 <= int(s) <= len(TIMEFRAMES)]
            else:
                console.print("[red]Invalid choice[/red]")
                selected_timeframes = []

            if selected_timeframes:
                output_dir = Path("data/benchmark")
                output_dir.mkdir(parents=True, exist_ok=True)

                console.print(f"\n[bold]Generating benchmarks for: {', '.join(selected_timeframes)}[/bold]\n")

                # Generate Binance benchmarks
                if exchange_choice in ['1', '3']:
                    console.print("\n[bold cyan]===== Binance CMC Top 10 Index =====[/bold cyan]")
                    for tf in selected_timeframes:
                        create_benchmark_candles(data_dir, tf, output_dir)

                # Generate Upbit benchmarks
                if exchange_choice in ['2', '3']:
                    console.print("\n[bold magenta]===== Upbit Composite Index =====[/bold magenta]")
                    for tf in selected_timeframes:
                        create_upbit_benchmark_candles(data_dir, tf, output_dir)

                console.print("\n[bold green]✓ Benchmark generation complete![/bold green]")
                console.print(f"[green]Output directory: {output_dir}[/green]")