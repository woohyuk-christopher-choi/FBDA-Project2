#!/usr/bin/env python3
"""
데이터 수집 클래스 - 바이낸스 및 업비트
"""

import time
import polars as pl
from datetime import datetime
from typing import Optional, List, Tuple

from src.api import BinanceAPIClient, UpbitAPIClient
from src.data import BinanceDataProcessor, UpbitDataProcessor, DataStorage
from src.utils import UIManager


class BinanceTradeCollector:
    """바이낸스 거래 데이터 수집 클래스"""

    def __init__(self, output_dir: str = "data"):
        self.client = BinanceAPIClient()
        self.processor = BinanceDataProcessor()
        self.storage = DataStorage(output_dir)
        self.ui = UIManager()

    def collect_trades(
            self,
            symbol: str,
            start_time: Optional[int] = None,
            end_time: Optional[int] = None,
            max_trades: Optional[int] = None,
            save_format: str = "arrow"
    ) -> Optional[pl.DataFrame]:
        """
        바이낸스 거래 데이터 수집
        
        Args:
            symbol: 거래쌍 (예: "BTCUSDT")
            start_time: 시작 시간 (밀리초, None이면 24시간 전)
            end_time: 종료 시간 (밀리초, None이면 현재)
            max_trades: 최대 수집 개수
            save_format: 저장 형식 ("arrow", "parquet", "csv")
            
        Returns:
            수집된 Polars DataFrame
        """
        # 기본값 설정
        if start_time is None:
            start_time = int((time.time() - 86400) * 1000)  # 24시간 전
        if end_time is None:
            end_time = int(time.time() * 1000)  # 현재

        # 기존 데이터 확인
        filepath = self.storage.get_filepath("binance", symbol, save_format)
        existing_data = self.storage.load_existing_data(filepath, save_format)
        
        existing_df = None
        existing_start = None
        existing_end = None
        
        if existing_data:
            existing_df, existing_start, existing_end = existing_data

        # 수집 범위 분석
        collect_ranges = self._analyze_gaps(
            start_time, end_time, existing_start, existing_end
        )
        
        if not collect_ranges:
            self.ui.show_warning("✓ 요청한 범위는 이미 모두 수집되어 있습니다.")
            return existing_df

        # 거래쌍 정보 확인
        symbol_info = self.client.get_exchange_info(symbol)
        if symbol_info is None:
            self.ui.show_error(f"✗ 유효하지 않은 거래쌍: {symbol}")
            return None

        # 각 범위별로 데이터 수집
        all_collected = []
        
        for range_start, range_end, range_label in collect_ranges:
            df = self._collect_range(
                symbol, range_start, range_end, range_label, 
                max_trades, save_format
            )
            if df is not None:
                all_collected.append(df)

        if not all_collected:
            self.ui.show_warning("수집된 데이터가 없습니다.")
            return existing_df if existing_df is not None else None

        # 데이터 병합
        final_df = self._merge_data(existing_df, all_collected, collect_ranges)

        # 통계 출력 및 저장
        self.ui.show_statistics(final_df, symbol, "바이낸스")
        self.storage.save_data(final_df, filepath, save_format, 'agg_trade_id')

        return final_df

    def _analyze_gaps(
            self,
            start_time: int,
            end_time: int,
            existing_start: Optional[int],
            existing_end: Optional[int]
    ) -> List[Tuple[int, int, str]]:
        """수집 범위 분석 - 요청한 범위만 수집"""
        collect_ranges = []
        
        if existing_start is not None and existing_end is not None:
            self.ui.show_warning(
                f"요청 범위: {datetime.fromtimestamp(start_time/1000)} ~ "
                f"{datetime.fromtimestamp(end_time/1000)}"
            )
            
            # 요청 범위가 기존 데이터에 완전히 포함되면 수집 불필요
            if start_time >= existing_start and end_time <= existing_end:
                self.ui.show_info("→ 요청한 범위는 이미 수집되어 있습니다.")
                return collect_ranges
            
            # 요청 범위가 기존 데이터와 겹치면 겹치지 않는 부분만 수집
            actual_start = start_time
            actual_end = end_time
            
            # 요청 시작이 기존 데이터 범위 내에 있으면 조정
            if start_time >= existing_start and start_time <= existing_end:
                actual_start = existing_end + 1
            
            # 요청 종료가 기존 데이터 범위 내에 있으면 조정
            if end_time >= existing_start and end_time <= existing_end:
                actual_end = existing_start - 1
            
            # 조정된 범위가 유효한지 확인
            if actual_start <= actual_end:
                collect_ranges.append((actual_start, actual_end, "요청 범위"))
                self.ui.show_info(
                    f"→ 수집 범위: {datetime.fromtimestamp(actual_start/1000)} ~ "
                    f"{datetime.fromtimestamp(actual_end/1000)}"
                )
        else:
            # 기존 데이터가 없으면 요청 범위 전체 수집
            collect_ranges.append((start_time, end_time, "전체"))
        
        return collect_ranges

    def _collect_range(
            self,
            symbol: str,
            range_start: int,
            range_end: int,
            range_label: str,
            max_trades: Optional[int],
            save_format: str
    ) -> Optional[pl.DataFrame]:
        """특정 범위의 데이터 수집"""
        start_dt = datetime.fromtimestamp(range_start / 1000)
        end_dt = datetime.fromtimestamp(range_end / 1000)

        self.ui.show_collection_info(
            "바이낸스", symbol, start_dt, end_dt, save_format, range_label
        )

        all_trades = []
        current_time = range_start
        request_count = 0
        total_time_range = range_end - range_start

        progress = self.ui.create_binance_progress()

        with progress:
            task = progress.add_task(
                f"[bold green]수집 중: {symbol} ({range_label})",
                total=100,
                trades="0개",
                requests="0회",
                weight="N/A"
            )

            retry_count = 0
            max_retries = 5
            retry_delay = 5

            try:
                while True:
                    # 최대 개수 체크
                    if max_trades and len(all_trades) >= max_trades:
                        progress.update(task, description=f"[bold green]✓ 완료: {symbol} (최대 개수)")
                        break

                    try:
                        trades, status_code, headers = self.client.fetch_trades(
                            symbol, current_time, range_end
                        )

                        # 에러 처리
                        if status_code != 200:
                            if status_code == 429:
                                self.ui.show_warning("Rate limit 초과, 60초 대기...")
                                time.sleep(60)
                                continue
                            else:
                                self.ui.show_error(f"API 에러: {status_code}")
                                break

                        retry_count = 0

                    except Exception as e:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = retry_delay * retry_count
                            self.ui.show_warning(f"네트워크 오류 ({retry_count}/{max_retries}): {e}")
                            self.ui.show_warning(f"{wait_time}초 후 재시도...")
                            time.sleep(wait_time)
                            continue
                        else:
                            self.ui.show_error("최대 재시도 횟수 초과. 수집을 중단합니다.")
                            break

                    # 데이터 없음
                    if not trades:
                        progress.update(task, description=f"[bold green]✓ 완료: {symbol}")
                        break

                    all_trades.extend(trades)
                    request_count += 1

                    # 진행률 업데이트
                    time_progress = ((current_time - range_start) / total_time_range) * 100
                    weight_used = headers.get('X-MBX-USED-WEIGHT-1M', 'N/A')

                    progress.update(
                        task,
                        completed=min(time_progress, 100),
                        trades=f"{len(all_trades):,}개",
                        requests=f"{request_count}회",
                        weight=f"Weight: {weight_used}"
                    )

                    # 다음 시작 시간
                    current_time = trades[-1]['T'] + 1

                    # 종료 조건
                    if current_time >= range_end:
                        progress.update(task, description=f"[bold green]✓ 완료: {symbol}")
                        break

                    time.sleep(0.05)  # Rate limit 방지

            except KeyboardInterrupt:
                progress.update(task, description=f"[bold yellow]⚠ 중단됨: {symbol}")
                self.ui.show_warning("사용자에 의해 중단되었습니다.")
            except Exception as e:
                progress.update(task, description=f"[bold red]✗ 에러: {symbol}")
                self.ui.show_error(f"데이터 수집 중 에러: {e}")

        self.ui.show_completion(len(all_trades), request_count, range_label)

        if all_trades:
            return self.processor.convert_to_dataframe(all_trades)
        return None

    def _merge_data(
            self,
            existing_df: Optional[pl.DataFrame],
            collected_dfs: List[pl.DataFrame],
            collect_ranges: List[Tuple]
    ) -> pl.DataFrame:
        """데이터 병합"""
        if existing_df is not None:
            result_dfs = []
            
            # 앞쪽 gap 먼저
            for gap_start, gap_end, gap_label in collect_ranges:
                if gap_label == "앞쪽 감" and collected_dfs:
                    result_dfs.append(collected_dfs.pop(0))
            
            result_dfs.append(existing_df)
            result_dfs.extend(collected_dfs)
            
            final_df = pl.concat(result_dfs)
        else:
            final_df = pl.concat(collected_dfs)
        
        # 중복 제거 및 정렬
        final_df = final_df.unique(subset=['agg_trade_id'], keep='last')
        final_df = final_df.sort('timestamp')
        
        return final_df


class UpbitTradeCollector:
    """업비트 거래 데이터 수집 클래스"""

    def __init__(self, output_dir: str = "data"):
        self.client = UpbitAPIClient()
        self.processor = UpbitDataProcessor()
        self.storage = DataStorage(output_dir)
        self.ui = UIManager()

    def collect_trades(
            self,
            market: str,
            max_trades: int = 10000,
            save_format: str = "arrow"
    ) -> Optional[pl.DataFrame]:
        """
        업비트 거래 데이터 수집
        
        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            max_trades: 최대 수집 개수
            save_format: 저장 형식 ("arrow", "parquet", "csv")
            
        Returns:
            수집된 Polars DataFrame
        """
        # 기존 데이터 확인
        filepath = self.storage.get_filepath("upbit", market, save_format)
        existing_data = self.storage.load_existing_data(filepath, save_format)
        
        last_seq_id = None
        if existing_data:
            existing_df, _, _ = existing_data
            last_seq_id = existing_df['sequential_id'].min()  # 업비트는 역순
            self.ui.show_info(
                f"기존 데이터 이후부터 수집 시작 (last_seq_id: {last_seq_id})"
            )

        # 마켓 정보 확인
        market_info = self.client.get_market_info(market)
        if market_info is None:
            self.ui.show_error(f"✗ 유효하지 않은 마켓: {market}")
            return None

        self.ui.show_collection_info("업비트", market, save_format=save_format)

        all_trades = []
        cursor = None
        request_count = 0

        progress = self.ui.create_upbit_progress(max_trades)

        with progress:
            task = progress.add_task(
                f"[bold cyan]수집 중: {market}",
                total=max_trades,
                current="0",
                target=f"{max_trades:,}",
                requests="0회",
                latest="N/A"
            )

            retry_count = 0
            max_retries = 5
            retry_delay = 5

            try:
                while len(all_trades) < max_trades:
                    count = min(500, max_trades - len(all_trades))

                    try:
                        trades, status_code, _ = self.client.fetch_trades(
                            market, count, cursor
                        )

                        if status_code != 200:
                            if status_code == 429:
                                self.ui.show_warning("Rate limit 초과, 60초 대기...")
                                time.sleep(60)
                                continue
                            else:
                                self.ui.show_error(f"API 에러: {status_code}")
                                break

                        retry_count = 0

                    except Exception as e:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = retry_delay * retry_count
                            self.ui.show_warning(f"네트워크 오류 ({retry_count}/{max_retries}): {e}")
                            self.ui.show_warning(f"{wait_time}초 후 재시도...")
                            time.sleep(wait_time)
                            continue
                        else:
                            self.ui.show_error("최대 재시도 횟수 초과.")
                            break

                    if not trades:
                        progress.update(task, description=f"[bold cyan]✓ 완료: {market}")
                        break

                    # 기존 데이터와 중복 확인
                    new_trades = []
                    for trade in trades:
                        if last_seq_id is None or trade['sequential_id'] < last_seq_id:
                            new_trades.append(trade)
                        else:
                            progress.update(task, description=f"[bold cyan]✓ 완료: {market} (기존 데이터 도달)")
                            all_trades.extend(new_trades)
                            break
                    else:
                        all_trades.extend(new_trades)
                        request_count += 1

                    # 기존 데이터 도달 시 종료
                    if new_trades and last_seq_id and min(t['sequential_id'] for t in new_trades) >= last_seq_id:
                        break

                    if not new_trades:
                        break

                    # 진행률 업데이트
                    progress.update(
                        task,
                        completed=len(all_trades),
                        current=f"{len(all_trades):,}",
                        target=f"{max_trades:,}",
                        requests=f"{request_count}회",
                        latest=f"{trades[0]['trade_date_utc'][:10]}" if trades else "N/A"
                    )

                    cursor = trades[-1]['sequential_id']

                    if len(all_trades) >= max_trades:
                        progress.update(task, description=f"[bold cyan]✓ 완료: {market}")
                        break

                    time.sleep(0.1)  # Rate limit 방지

            except KeyboardInterrupt:
                progress.update(task, description=f"[bold yellow]⚠ 중단됨: {market}")
                self.ui.show_warning("사용자에 의해 중단되었습니다.")
            except Exception as e:
                progress.update(task, description=f"[bold red]✗ 에러: {market}")
                self.ui.show_error(f"데이터 수집 중 에러: {e}")

        self.ui.show_completion(len(all_trades), request_count)

        if not all_trades:
            self.ui.show_warning("수집된 데이터가 없습니다.")
            return None

        df = self.processor.convert_to_dataframe(all_trades)
        self.ui.show_statistics(df, market, "업비트")
        self.storage.save_data(df, filepath, save_format, 'sequential_id')

        return df
