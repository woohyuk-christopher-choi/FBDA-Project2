# frequency_model/modeling.py

from portfolio import run_real_data_analysis
from typing import Optional, List

def run_analysis_for_exchange(
    exchange_name: str, 
    start_date: Optional[str], 
    end_date: Optional[str],
    use_tarv: bool = True,
    n_portfolios: int = 3,
    window_days: int = 30,
    min_observations: int = 10000
):
    """
    지정된 거래소 데이터에 대해 Low-Beta Anomaly 분석을 실행합니다.
    portfolio.py의 run_real_data_analysis 함수를 호출하여 전체 프로세스를 수행합니다.
    """
    print(f"\n{'='*30} {exchange_name.upper()} ANALYSIS START {'='*30}")
    
    try:
        # portfolio.py의 run_real_data_analysis 함수를 사용하여
        # 데이터 로딩, 베타 추정, 포트폴리오 구성 및 백테스팅을 한 번에 실행합니다.
        run_real_data_analysis(
            data_dir="data",
            exchange=exchange_name,
            start_date=start_date,
            end_date=end_date,
            use_tarv=use_tarv,
            n_portfolios=n_portfolios,
            window_days=window_days,
            min_observations=min_observations
        )

    except ValueError as e:
        print(f"\n❌ {exchange_name.upper()} 분석 중 오류 발생: {e}")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    # 분석할 기간을 설정합니다.
    # None으로 설정하면 해당 거래소의 전체 기간 데이터를 사용합니다.
    START_DATE = "2024-01-01"
    END_DATE = "2024-06-30"
    
    # 1. Binance 데이터로 모델링 실행
    run_analysis_for_exchange("binance", START_DATE, END_DATE)
    
    # 2. Upbit 데이터로 모델링 실행
    # run_analysis_for_exchange("upbit", START_DATE, END_DATE)

    print(f"\n{'='*30} ALL ANALYSES COMPLETED {'='*30}")