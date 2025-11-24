#!/usr/bin/env python3
"""
UI 모듈 - Rich 콘솔 출력 및 진행 표시
"""

import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import box
from datetime import datetime
from typing import Optional

console = Console()


class UIManager:
    """UI 및 진행 표시 관리 클래스"""

    @staticmethod
    def show_header(title: str, subtitle: str = ""):
        """헤더 출력"""
        console.print("\n")
        text = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            text += f"\n[dim]{subtitle}[/dim]"
        console.print(Panel.fit(text, border_style="cyan"))

    @staticmethod
    def show_collection_info(
            exchange: str,
            symbol: str,
            start_dt: Optional[datetime] = None,
            end_dt: Optional[datetime] = None,
            save_format: str = "arrow",
            gap_label: str = "전체"
    ):
        """데이터 수집 정보 패널"""
        text = f"[bold cyan]🚀 {exchange} 데이터 수집 ({gap_label})[/bold cyan]\n\n"
        text += f"[yellow]{'마켓' if exchange == '업비트' else '거래쌍'}:[/yellow] {symbol}\n"
        
        if start_dt:
            text += f"[yellow]시작:[/yellow] {start_dt}\n"
        if end_dt:
            text += f"[yellow]종료:[/yellow] {end_dt}\n"
            
        text += f"[yellow]저장 형식:[/yellow] {save_format.upper()}"
        
        console.print(Panel.fit(text, border_style="cyan"))

    @staticmethod
    def create_binance_progress() -> Progress:
        """바이낸스용 Progress 객체 생성"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[trades]}[/cyan]"),
            TextColumn("•"),
            TextColumn("[yellow]{task.fields[requests]}[/yellow]"),
            TextColumn("•"),
            TextColumn("[magenta]{task.fields[weight]}[/magenta]"),
            TimeElapsedColumn(),
            console=console,
            expand=True
        )

    @staticmethod
    def create_upbit_progress(max_trades: int) -> Progress:
        """업비트용 Progress 객체 생성"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="bold cyan"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[current]}/{task.fields[target]}[/cyan]"),
            TextColumn("•"),
            TextColumn("[yellow]{task.fields[requests]}[/yellow]"),
            TextColumn("•"),
            TextColumn("[magenta]{task.fields[latest]}[/magenta]"),
            TimeElapsedColumn(),
            console=console,
            expand=True
        )

    @staticmethod
    def show_statistics(df: pl.DataFrame, symbol: str, exchange: str = "바이낸스"):
        """데이터 통계 테이블 출력"""
        table = Table(
            title=f"📊 {symbol} 통계",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("항목", style="cyan", no_wrap=True)
        table.add_column("값", style="green")

        table.add_row("기간", f"{df['datetime'].min()} ~ {df['datetime'].max()}")
        table.add_row("데이터 건수", f"{len(df):,}개")

        # 데이터 타입 확인 (캔들 vs 거래)
        is_klines = 'open' in df.columns and 'close' in df.columns
        
        # 통화 기호
        currency = "$" if exchange == "바이낸스" else "₩"
        
        if is_klines:
            # 캔들 데이터 통계
            if exchange == "바이낸스":
                table.add_row("시가 평균", f"{currency}{df['open'].mean():,.2f}")
                table.add_row("최고가", f"{currency}{df['high'].max():,.2f}")
                table.add_row("최저가", f"{currency}{df['low'].min():,.2f}")
                table.add_row("종가 평균", f"{currency}{df['close'].mean():,.2f}")
                table.add_row("총 거래량", f"{df['volume'].sum():,.4f}")
                if 'quote_volume' in df.columns:
                    table.add_row("총 거래 금액", f"{currency}{df['quote_volume'].sum():,.2f}")
            else:
                table.add_row("시가 평균", f"{currency}{df['open'].mean():,.0f}")
                table.add_row("최고가", f"{currency}{df['high'].max():,.0f}")
                table.add_row("최저가", f"{currency}{df['low'].min():,.0f}")
                table.add_row("종가 평균", f"{currency}{df['close'].mean():,.0f}")
                table.add_row("총 거래량", f"{df['volume'].sum():,.4f}")
                if 'quote_volume' in df.columns:
                    table.add_row("총 거래 금액", f"{currency}{df['quote_volume'].sum():,.0f}")
        else:
            # 거래 데이터 통계
            price_col = 'price' if 'price' in df.columns else 'trade_price'
            volume_col = 'quantity' if 'quantity' in df.columns else 'trade_volume'
            
            if exchange == "바이낸스":
                table.add_row("평균 가격", f"{currency}{df[price_col].mean():,.2f}")
                table.add_row("최고가", f"{currency}{df[price_col].max():,.2f}")
                table.add_row("최저가", f"{currency}{df[price_col].min():,.2f}")
                table.add_row("총 거래량", f"{df[volume_col].sum():,.4f}")
                if 'amount' in df.columns:
                    table.add_row("총 거래 금액", f"{currency}{df['amount'].sum():,.2f}")
            else:
                table.add_row("평균 가격", f"{currency}{df[price_col].mean():,.0f}")
                table.add_row("최고가", f"{currency}{df[price_col].max():,.0f}")
                table.add_row("최저가", f"{currency}{df[price_col].min():,.0f}")
                table.add_row("총 거래량", f"{df[volume_col].sum():,.4f}")
                if 'amount' in df.columns:
                    table.add_row("총 거래 금액", f"{currency}{df['amount'].sum():,.0f}")

            # 매수/매도 통계 (거래 데이터만)
            if 'side' in df.columns:
                side_counts = df.group_by('side').count()
                for row in side_counts.iter_rows(named=True):
                    table.add_row(f"{row['side']} 거래", f"{row['count']:,}건")

        console.print(table)

    @staticmethod
    def show_completion(trade_count: int, request_count: int, gap_label: str = ""):
        """수집 완료 메시지"""
        label = f" ({gap_label})" if gap_label else ""
        console.print(f"\n[bold green]✓ 수집 완료{label}![/bold green]")
        console.print(f"  총 거래: [cyan]{trade_count:,}개[/cyan]")
        console.print(f"  총 요청: [yellow]{request_count}회[/yellow]\n")

    @staticmethod
    def show_error(message: str):
        """에러 메시지"""
        console.print(f"[red]{message}[/red]")

    @staticmethod
    def show_warning(message: str):
        """경고 메시지"""
        console.print(f"[yellow]{message}[/yellow]")

    @staticmethod
    def show_info(message: str):
        """정보 메시지"""
        console.print(f"[cyan]{message}[/cyan]")

    @staticmethod
    def get_input(prompt: str) -> str:
        """사용자 입력"""
        return console.input(f"[bold yellow]{prompt}[/bold yellow] ").strip()
