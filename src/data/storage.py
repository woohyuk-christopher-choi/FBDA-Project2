#!/usr/bin/env python3
"""
저장소 모듈 - 파일 I/O 및 데이터 저장
"""

import polars as pl
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

console = Console()


class DataStorage:
    """데이터 저장 및 로드 클래스"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_filepath(self, exchange: str, symbol: str, format: str, data_type: str = "trades") -> Path:
        """파일 경로 생성"""
        symbol_clean = symbol.replace('-', '_')
        
        extension_map = {
            "arrow": "arrow",
            "parquet": "parquet",
            "csv": "csv"
        }
        
        ext = extension_map.get(format, "arrow")
        return self.output_dir / f"{exchange}_{symbol_clean}_{data_type}.{ext}"

    def load_existing_data(
            self,
            filepath: Path,
            format: str
    ) -> Optional[Tuple[pl.DataFrame, int, int]]:
        """
        기존 데이터 로드
        
        Returns:
            (DataFrame, min_timestamp, max_timestamp) 또는 None
        """
        if not filepath.exists():
            return None

        try:
            if format == "arrow":
                df = pl.read_ipc(filepath, memory_map=False)
            elif format == "parquet":
                df = pl.read_parquet(filepath, memory_map=False)
            elif format == "csv":
                df = pl.read_csv(filepath)
            else:
                console.print(f"[red]지원하지 않는 형식: {format}[/red]")
                return None

            # datetime 컬럼이 없으면 timestamp로부터 생성 (기존 데이터 호환성)
            if 'datetime' not in df.columns:
                df = df.with_columns([
                    pl.from_epoch(pl.col('timestamp'), time_unit='ms').alias('datetime')
                ])
            
            min_ts = df['timestamp'].min()
            max_ts = df['timestamp'].max()

            console.print(f"[cyan]기존 데이터 발견: {len(df):,}개[/cyan]")
            console.print(f"[cyan]기존 범위: {df['datetime'].min()} ~ {df['datetime'].max()}[/cyan]")

            return df, min_ts, max_ts

        except Exception as e:
            console.print(f"[yellow]기존 파일 읽기 실패: {e}[/yellow]")
            return None

    def save_data(
            self,
            df: pl.DataFrame,
            filepath: Path,
            format: str,
            id_column: str
    ) -> None:
        """
        데이터 저장 (기존 파일이 있으면 병합)
        
        Args:
            df: 저장할 DataFrame
            filepath: 저장 경로
            format: 저장 형식
            id_column: 중복 제거 기준 컬럼
        """
        with console.status(f"[bold green]저장 중...", spinner="dots"):
            # 기존 파일이 있으면 병합
            if filepath.exists():
                try:
                    if format == "arrow":
                        existing_df = pl.read_ipc(filepath, memory_map=False)
                    elif format == "parquet":
                        existing_df = pl.read_parquet(filepath, memory_map=False)
                    elif format == "csv":
                        existing_df = pl.read_csv(filepath)
                    else:
                        console.print(f"[red]지원하지 않는 형식: {format}[/red]")
                        return

                    old_count = len(existing_df)
                    df = pl.concat([existing_df, df])
                    df = df.unique(subset=[id_column], keep='last')
                    df = df.sort('timestamp')
                    new_count = len(df) - old_count

                    console.print(
                        f"[cyan]합치기: {old_count:,}개 + {new_count:,}개 (새 데이터) = {len(df):,}개[/cyan]"
                    )
                    del existing_df

                except Exception as e:
                    console.print(f"[yellow]기존 파일 병합 실패, 새로 저장: {e}[/yellow]")

            # 파일 저장
            if format == "arrow":
                df.write_ipc(filepath)
            elif format == "parquet":
                df.write_parquet(filepath, compression="snappy")
            elif format == "csv":
                df.write_csv(filepath)

        # 저장 완료 메시지
        file_size = filepath.stat().st_size / 1024 / 1024
        console.print(f"[bold green]✓ 저장 완료![/bold green]")
        console.print(f"  파일: [cyan]{filepath}[/cyan]")
        console.print(f"  크기: [yellow]{file_size:.2f} MB[/yellow]\n")
