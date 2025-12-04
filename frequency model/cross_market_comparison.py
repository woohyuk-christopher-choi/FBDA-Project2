#!/usr/bin/env python3
"""
Cross-Market Comparison: Equities vs. Crypto
Low-Volatility Anomaly Analysis

Motivation:
1. Equities and crypto differ fundamentally in:
   - Market microstructure
   - Tail behavior
   - Liquidity
   - Efficiency
2. TARV reveals how these structural differences affect volatility-return patterns

Analysis:
- Apply same methodology to S&P500 stocks and crypto (Binance/Upbit)
- Use universal alpha estimation across markets
- Compare TARV vs Standard RV results

Interpretation:
1. If equities show rational high-risk-high-return under TARV
   → Anomaly may diminish under better volatility measurement
2. If crypto preserves or amplifies the anomaly
   → Reflects microstructure inefficiency or behavioral dynamics
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

# Import data loaders
from stock_data_loader import load_sp500_data, resample_to_frequency
from data_loader import prepare_data_for_analysis, FREQUENCY_MINUTES

# Import portfolio and TARV tools
from portfolio import BetaEstimator, PortfolioConstructor, BacktestEngine
from tarv import estimate_universal_alpha


# =============================================================================
# Configuration
# =============================================================================

FREQUENCIES = ['1m', '5m', '15m', '30m', '1h', '1d']

# Market-specific configurations
MARKET_CONFIGS = {
    'stocks': {
        'name': 'S&P500 Stocks',
        'market_hours_per_day': 390,  # 6.5 hours * 60 min
        'frequency_params': {
            # 1m: 더 짧은 window + 약간 높은 min_obs_ratio (데이터 품질 확보)
            '1m':  {'window_days': 10,  'min_obs_ratio': 0.4},
            '5m':  {'window_days': 20,  'min_obs_ratio': 0.3},
            '15m': {'window_days': 30,  'min_obs_ratio': 0.3},
            '30m': {'window_days': 30,  'min_obs_ratio': 0.3},
            '1h':  {'window_days': 60,  'min_obs_ratio': 0.3},
            '1d':  {'window_days': 120, 'min_obs_ratio': 0.3}
        }
    },
    'crypto': {
        'name': 'Cryptocurrency',
        'market_hours_per_day': 1440,  # 24 hours * 60 min
        'frequency_params': {
            # 1m: 24h 거래 + 풍부한 데이터 → window_days는 20일 정도, ratio는 0.3
            '1m':  {'window_days': 20,  'min_obs_ratio': 0.3},
            '5m':  {'window_days': 30,  'min_obs_ratio': 0.2},
            '15m': {'window_days': 30,  'min_obs_ratio': 0.25},
            '30m': {'window_days': 30,  'min_obs_ratio': 0.25},
            '1h':  {'window_days': 60,  'min_obs_ratio': 0.33},
            '1d':  {'window_days': 180, 'min_obs_ratio': 0.33}
        }
    }
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_stock_market_data(
    data_dir: Path,
    frequency: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load S&P500 stock data."""
    print(f"\n📊 Loading Stock Market Data ({frequency})")
    print("-" * 70)

    stock_file = data_dir / "S&P500_index포함_분봉_20241126_20251126.xlsx"

    if not stock_file.exists():
        raise FileNotFoundError(f"Stock data file not found: {stock_file}")

    # Load data
    stock_prices, market_prices = load_sp500_data(
        str(stock_file),
        start_date=start_date,
        end_date=end_date
    )

    # Resample if needed
    if frequency != '1m':
        freq_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '1d': '1d'}
        resample_freq = freq_map.get(frequency, frequency)

        stock_prices = resample_to_frequency(stock_prices, resample_freq)
        market_prices = resample_to_frequency(
            pd.DataFrame({'market': market_prices}),
            resample_freq
        )['market']

    print(f"✅ Loaded: {len(stock_prices.columns)} stocks, {len(stock_prices):,} observations")
    print(f"   Date range: {stock_prices.index[0]} to {stock_prices.index[-1]}")

    return stock_prices, market_prices


def load_crypto_market_data(
    data_dir: Path,
    exchange: str,
    frequency: str,
    start_date: str,
    end_date: str,
    min_observations: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load crypto market data."""
    print(f"\n📊 Loading Crypto Market Data ({exchange} - {frequency})")
    print("-" * 70)

    assets_prices, market_prices = prepare_data_for_analysis(
        data_dir=str(data_dir),
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        min_observations=min_observations,
        frequency=frequency
    )

    print(f"✅ Loaded: {len(assets_prices.columns)} assets, {len(assets_prices):,} observations")

    return assets_prices, market_prices


# =============================================================================
# Universal Alpha Estimation
# =============================================================================

def estimate_cross_market_alpha(
    stock_prices: pd.DataFrame,
    stock_market: pd.Series,
    crypto_prices: pd.DataFrame,
    crypto_market: pd.Series,
    K: int,
    method: str = 'median'
) -> Tuple[float, Dict]:
    """
    Estimate universal alpha across stock and crypto markets.

    Implements professor's suggestion:
    "stock이랑 crypto 둘다 사용해서 universal level alpha를 사용하는게 더 좋을 수 있다"

    Args:
        stock_prices: Stock price DataFrame
        stock_market: Stock market index
        crypto_prices: Crypto price DataFrame
        crypto_market: Crypto market index
        K: Pre-averaging window size
        method: 'median', 'mean', or 'min' for combining alphas

    Returns:
        Tuple of (universal_alpha, info_dict)
    """
    print(f"\n🌍 Estimating Universal Alpha Across Markets")
    print("-" * 70)
    print(f"   Method: {method}")
    print(f"   Pre-averaging window K: {K}")

    # Combine all return series from both markets
    returns_list = []

    # Stock returns
    stock_returns = stock_prices.pct_change().dropna()
    stock_market_returns = stock_market.pct_change().dropna()

    for col in stock_returns.columns[:10]:  # Sample 10 stocks
        returns_list.append(stock_returns[col].dropna().values)
    returns_list.append(stock_market_returns.dropna().values)

    # Crypto returns
    crypto_returns = crypto_prices.pct_change().dropna()
    crypto_market_returns = crypto_market.pct_change().dropna()

    for col in crypto_returns.columns[:10]:  # Sample 10 cryptos
        returns_list.append(crypto_returns[col].dropna().values)
    returns_list.append(crypto_market_returns.dropna().values)

    # Estimate universal alpha
    universal_alpha, info = estimate_universal_alpha(
        returns_list=returns_list,
        K=K,
        p=2,
        method=method
    )

    print(f"\n✅ Universal Alpha: {universal_alpha:.3f}")
    print(f"   Based on {info['n_series']} series")
    print(f"   Individual alphas: {[f'{a:.3f}' for a in info['individual_alphas'][:5]]}...")

    return universal_alpha, info


# =============================================================================
# Single Market Analysis
# =============================================================================

def run_single_market_analysis(
    market_type: str,
    assets_prices: pd.DataFrame,
    market_prices: pd.Series,
    frequency: str,
    n_portfolios: int = 3,
    universal_alpha: Optional[float] = None,
    verbose: bool = False
) -> Dict:
    """
    Run analysis for a single market with both TARV and Standard RV.

    Args:
        market_type: 'stocks' or 'crypto'
        assets_prices: Asset price DataFrame
        market_prices: Market benchmark Series
        frequency: Frequency string
        n_portfolios: Number of portfolios
        universal_alpha: If provided, use this alpha for TARV estimation
        verbose: Print detailed output

    Returns:
        Dict with 'tarv_results', 'standard_results', 'metadata'
    """
    config = MARKET_CONFIGS[market_type]
    freq_params = config['frequency_params'][frequency]

    # Calculate observations per day
    freq_minutes = FREQUENCY_MINUTES.get(frequency, 1)

    if frequency == '1d':
        obs_per_day = 1
    else:
        obs_per_day = config['market_hours_per_day'] // freq_minutes

    window_days = freq_params['window_days']
    window_obs = obs_per_day * window_days
    min_obs = int(window_obs * freq_params['min_obs_ratio'])

    print(f"\n{'='*70}")
    print(f"📊 MARKET: {config['name']} | FREQUENCY: {frequency}")
    print(f"{'='*70}")
    print(f"   Observations per day: {obs_per_day}")
    print(f"   Window: {window_days} days ({window_obs} obs)")
    print(f"   Min observations: {min_obs}")
    if universal_alpha:
        print(f"   Using Universal Alpha: {universal_alpha:.3f}")

    results = {
        'market_type': market_type,
        'frequency': frequency,
        'metadata': {
            'n_assets': len(assets_prices.columns),
            'n_observations': len(assets_prices),
            'date_range': (str(assets_prices.index[0]), str(assets_prices.index[-1])),
            'window_days': window_days,
            'obs_per_day': obs_per_day
        }
    }

    # Portfolio constructor (same for both methods)
    portfolio_constructor = PortfolioConstructor(
        n_portfolios=n_portfolios,
        weighting='equal',
        long_short=True
    )

    # Method 1: TARV
    print("\n" + "-" * 70)
    print("📊 METHOD 1: TARV (Tail-Adaptive RV)")
    print("-" * 70)

    tarv_estimator = BetaEstimator(
        window_observations=window_obs,
        min_observations=min_obs,
        use_tarv=True,
        c_omega=0.333,
        use_rolling_alpha=False,
        verbose=verbose
    )

    # If universal_alpha provided, we'd need to modify BetaEstimator to accept it
    # For now, it will estimate individual alphas

    tarv_engine = BacktestEngine(
        beta_estimator=tarv_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency='weekly',
        transaction_cost=0.001
    )

    tarv_results = tarv_engine.run_backtest(
        assets_prices=assets_prices.copy(),
        market_prices=market_prices.copy()
    )
    results['tarv_results'] = tarv_results

    # Method 2: Standard RV
    print("\n" + "-" * 70)
    print("📊 METHOD 2: Standard RV")
    print("-" * 70)

    standard_estimator = BetaEstimator(
        window_observations=window_obs,
        min_observations=min_obs,
        use_tarv=False,
        verbose=verbose
    )

    standard_engine = BacktestEngine(
        beta_estimator=standard_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency='weekly',
        transaction_cost=0.001
    )

    standard_results = standard_engine.run_backtest(
        assets_prices=assets_prices.copy(),
        market_prices=market_prices.copy()
    )
    results['standard_results'] = standard_results

    # Calculate summary statistics
    results['summary'] = calculate_market_summary(tarv_results, standard_results)

    return results


def calculate_market_summary(
    tarv_results: Dict[str, pd.DataFrame],
    standard_results: Dict[str, pd.DataFrame]
) -> Dict:
    """Calculate summary statistics for a single market."""
    summary = {'tarv': {}, 'standard': {}}

    # TARV summary
    if 'Low_Beta' in tarv_results and 'High_Beta' in tarv_results:
        low_ret = tarv_results['Low_Beta']['return']
        high_ret = tarv_results['High_Beta']['return']
        common_idx = low_ret.index.intersection(high_ret.index)
        spread = low_ret.loc[common_idx] - high_ret.loc[common_idx]

        t_stat, p_val_two = stats.ttest_1samp(spread, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

        summary['tarv']['spread_mean'] = spread.mean()
        summary['tarv']['spread_std'] = spread.std()
        summary['tarv']['t_stat'] = t_stat
        summary['tarv']['p_value'] = p_val
        summary['tarv']['low_beta_mean'] = low_ret.mean()
        summary['tarv']['high_beta_mean'] = high_ret.mean()

    if 'BAB' in tarv_results:
        bab_ret = tarv_results['BAB']['return']
        t_stat, p_val_two = stats.ttest_1samp(bab_ret, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

        summary['tarv']['bab_mean'] = bab_ret.mean()
        summary['tarv']['bab_std'] = bab_ret.std()
        summary['tarv']['bab_sharpe'] = bab_ret.mean() / bab_ret.std() if bab_ret.std() > 0 else 0
        summary['tarv']['bab_t_stat'] = t_stat
        summary['tarv']['bab_p_value'] = p_val

    # Standard RV summary
    if 'Low_Beta' in standard_results and 'High_Beta' in standard_results:
        low_ret = standard_results['Low_Beta']['return']
        high_ret = standard_results['High_Beta']['return']
        common_idx = low_ret.index.intersection(high_ret.index)
        spread = low_ret.loc[common_idx] - high_ret.loc[common_idx]

        t_stat, p_val_two = stats.ttest_1samp(spread, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

        summary['standard']['spread_mean'] = spread.mean()
        summary['standard']['spread_std'] = spread.std()
        summary['standard']['t_stat'] = t_stat
        summary['standard']['p_value'] = p_val
        summary['standard']['low_beta_mean'] = low_ret.mean()
        summary['standard']['high_beta_mean'] = high_ret.mean()

    if 'BAB' in standard_results:
        bab_ret = standard_results['BAB']['return']
        t_stat, p_val_two = stats.ttest_1samp(bab_ret, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

        summary['standard']['bab_mean'] = bab_ret.mean()
        summary['standard']['bab_std'] = bab_ret.std()
        summary['standard']['bab_sharpe'] = bab_ret.mean() / bab_ret.std() if bab_ret.std() > 0 else 0
        summary['standard']['bab_t_stat'] = t_stat
        summary['standard']['bab_p_value'] = p_val

    return summary


# =============================================================================
# Cross-Market Comparison
# =============================================================================

def run_cross_market_comparison(
    data_dir: Path,
    frequency: str,
    crypto_exchange: str = 'binance',
    start_date: str = '2024-11-27',
    end_date: str = '2025-11-26',
    n_portfolios: int = 3,
    use_universal_alpha: bool = True
) -> Dict:
    """
    Run cross-market comparison for a single frequency.

    Returns:
        Dict with 'stocks', 'crypto', 'comparison' keys
    """
    print("\n" + "=" * 80)
    print(f"🌍 CROSS-MARKET COMPARISON: {frequency}")
    print("=" * 80)

    # Load stock data
    stock_prices, stock_market = load_stock_market_data(
        data_dir=data_dir,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date
    )

    # Load crypto data
    crypto_config = MARKET_CONFIGS['crypto']['frequency_params'][frequency]
    freq_minutes = FREQUENCY_MINUTES.get(frequency, 1)
    obs_per_day = 1440 // freq_minutes if frequency != '1d' else 1
    min_obs = int(obs_per_day * crypto_config['window_days'] * crypto_config['min_obs_ratio'])

    crypto_prices, crypto_market = load_crypto_market_data(
        data_dir=data_dir,
        exchange=crypto_exchange,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        min_observations=min_obs
    )

    # Estimate universal alpha (optional)
    universal_alpha = None
    if use_universal_alpha:
        # Calculate K based on frequency
        K = max(10, int(np.sqrt(min(len(stock_prices), len(crypto_prices)))))
        universal_alpha, alpha_info = estimate_cross_market_alpha(
            stock_prices=stock_prices,
            stock_market=stock_market,
            crypto_prices=crypto_prices,
            crypto_market=crypto_market,
            K=K,
            method='median'
        )

    # Run analysis for stocks
    stock_results = run_single_market_analysis(
        market_type='stocks',
        assets_prices=stock_prices,
        market_prices=stock_market,
        frequency=frequency,
        n_portfolios=n_portfolios,
        universal_alpha=universal_alpha,
        verbose=False
    )

    # Run analysis for crypto
    crypto_results = run_single_market_analysis(
        market_type='crypto',
        assets_prices=crypto_prices,
        market_prices=crypto_market,
        frequency=frequency,
        n_portfolios=n_portfolios,
        universal_alpha=universal_alpha,
        verbose=False
    )

    # Compare results
    comparison = compare_markets(stock_results, crypto_results, frequency)

    return {
        'frequency': frequency,
        'stocks': stock_results,
        'crypto': crypto_results,
        'comparison': comparison,
        'universal_alpha': universal_alpha
    }


def compare_markets(stock_results: Dict, crypto_results: Dict, frequency: str) -> Dict:
    """Compare results between stock and crypto markets."""
    comparison = {
        'frequency': frequency,
        'stock_vs_crypto': {}
    }

    # Compare TARV results
    if 'tarv' in stock_results['summary'] and 'tarv' in crypto_results['summary']:
        stock_tarv = stock_results['summary']['tarv']
        crypto_tarv = crypto_results['summary']['tarv']

        comparison['stock_vs_crypto']['tarv'] = {
            'stock_bab_mean': stock_tarv.get('bab_mean', np.nan) * 100,
            'crypto_bab_mean': crypto_tarv.get('bab_mean', np.nan) * 100,
            'stock_bab_sharpe': stock_tarv.get('bab_sharpe', np.nan),
            'crypto_bab_sharpe': crypto_tarv.get('bab_sharpe', np.nan),
            'stock_bab_tstat': stock_tarv.get('bab_t_stat', np.nan),
            'crypto_bab_tstat': crypto_tarv.get('bab_t_stat', np.nan),
            'stock_significant': stock_tarv.get('bab_p_value', 1) < 0.05,
            'crypto_significant': crypto_tarv.get('bab_p_value', 1) < 0.05
        }

    # Compare Standard RV results
    if 'standard' in stock_results['summary'] and 'standard' in crypto_results['summary']:
        stock_std = stock_results['summary']['standard']
        crypto_std = crypto_results['summary']['standard']

        comparison['stock_vs_crypto']['standard'] = {
            'stock_bab_mean': stock_std.get('bab_mean', np.nan) * 100,
            'crypto_bab_mean': crypto_std.get('bab_mean', np.nan) * 100,
            'stock_bab_sharpe': stock_std.get('bab_sharpe', np.nan),
            'crypto_bab_sharpe': crypto_std.get('bab_sharpe', np.nan),
            'stock_bab_tstat': stock_std.get('bab_t_stat', np.nan),
            'crypto_bab_tstat': crypto_std.get('bab_t_stat', np.nan),
            'stock_significant': stock_std.get('bab_p_value', 1) < 0.05,
            'crypto_significant': crypto_std.get('bab_p_value', 1) < 0.05
        }

    return comparison


# =============================================================================
# Output and Interpretation
# =============================================================================

def print_cross_market_summary(all_results: List[Dict]):
    """Print comprehensive cross-market comparison."""
    print("\n" + "=" * 80)
    print("📊 CROSS-MARKET COMPARISON SUMMARY")
    print("=" * 80)

    # Table 1: TARV Results
    print("\n" + "-" * 80)
    print("TABLE 1: BAB Strategy Performance with TARV")
    print("-" * 80)
    print(f"{'Frequency':<12} {'Market':<12} {'Mean Return':<15} {'Sharpe':<10} "
          f"{'t-stat':<10} {'Sig':<8}")
    print("-" * 80)

    for result in all_results:
        freq = result['frequency']

        # Stocks
        if 'bab_mean' in result['stocks']['summary']['tarv']:
            mean = result['stocks']['summary']['tarv']['bab_mean'] * 100
            sharpe = result['stocks']['summary']['tarv']['bab_sharpe']
            t = result['stocks']['summary']['tarv']['bab_t_stat']
            p = result['stocks']['summary']['tarv']['bab_p_value']
            sig = get_significance_symbol(p)
            print(f"{freq:<12} {'Stocks':<12} {mean:>13.3f}% {sharpe:>9.3f} "
                  f"{t:>9.3f} {sig:<8}")

        # Crypto
        if 'bab_mean' in result['crypto']['summary']['tarv']:
            mean = result['crypto']['summary']['tarv']['bab_mean'] * 100
            sharpe = result['crypto']['summary']['tarv']['bab_sharpe']
            t = result['crypto']['summary']['tarv']['bab_t_stat']
            p = result['crypto']['summary']['tarv']['bab_p_value']
            sig = get_significance_symbol(p)
            print(f"{'':<12} {'Crypto':<12} {mean:>13.3f}% {sharpe:>9.3f} "
                  f"{t:>9.3f} {sig:<8}")

        print()

    # Table 2: Standard RV Results
    print("\n" + "-" * 80)
    print("TABLE 2: BAB Strategy Performance with Standard RV")
    print("-" * 80)
    print(f"{'Frequency':<12} {'Market':<12} {'Mean Return':<15} {'Sharpe':<10} "
          f"{'t-stat':<10} {'Sig':<8}")
    print("-" * 80)

    for result in all_results:
        freq = result['frequency']

        # Stocks
        if 'bab_mean' in result['stocks']['summary']['standard']:
            mean = result['stocks']['summary']['standard']['bab_mean'] * 100
            sharpe = result['stocks']['summary']['standard']['bab_sharpe']
            t = result['stocks']['summary']['standard']['bab_t_stat']
            p = result['stocks']['summary']['standard']['bab_p_value']
            sig = get_significance_symbol(p)
            print(f"{freq:<12} {'Stocks':<12} {mean:>13.3f}% {sharpe:>9.3f} "
                  f"{t:>9.3f} {sig:<8}")

        # Crypto
        if 'bab_mean' in result['crypto']['summary']['standard']:
            mean = result['crypto']['summary']['standard']['bab_mean'] * 100
            sharpe = result['crypto']['summary']['standard']['bab_sharpe']
            t = result['crypto']['summary']['standard']['bab_t_stat']
            p = result['crypto']['summary']['standard']['bab_p_value']
            sig = get_significance_symbol(p)
            print(f"{'':<12} {'Crypto':<12} {mean:>13.3f}% {sharpe:>9.3f} "
                  f"{t:>9.3f} {sig:<8}")

        print()


def print_interpretation(all_results: List[Dict]):
    """Print interpretation of cross-market results."""
    print("\n" + "=" * 80)
    print("📋 INTERPRETATION: EQUITIES VS. CRYPTO")
    print("=" * 80)

    print("\n1️⃣  TARV EFFECT ON ANOMALY")
    print("-" * 80)
    print("Question: Does TARV reduce the anomaly differently across markets?")
    print()

    for result in all_results:
        freq = result['frequency']

        # Calculate TARV vs Standard difference for each market
        stock_tarv = result['stocks']['summary']['tarv'].get('bab_mean', np.nan) * 100
        stock_std = result['stocks']['summary']['standard'].get('bab_mean', np.nan) * 100
        crypto_tarv = result['crypto']['summary']['tarv'].get('bab_mean', np.nan) * 100
        crypto_std = result['crypto']['summary']['standard'].get('bab_mean', np.nan) * 100

        stock_diff = stock_std - stock_tarv
        crypto_diff = crypto_std - crypto_tarv

        print(f"{freq}:")
        print(f"  Stocks:  Standard RV: {stock_std:>6.3f}% | TARV: {stock_tarv:>6.3f}% | "
              f"Diff: {stock_diff:>6.3f}%")
        print(f"  Crypto:  Standard RV: {crypto_std:>6.3f}% | TARV: {crypto_tarv:>6.3f}% | "
              f"Diff: {crypto_diff:>6.3f}%")
        print()

    print("\n2️⃣  MARKET EFFICIENCY COMPARISON")
    print("-" * 80)
    print("Question: Which market shows stronger low-volatility anomaly?")
    print()

    stock_sig_count = 0
    crypto_sig_count = 0

    for result in all_results:
        freq = result['frequency']

        stock_p = result['stocks']['summary']['tarv'].get('bab_p_value', 1)
        crypto_p = result['crypto']['summary']['tarv'].get('bab_p_value', 1)

        stock_sig = stock_p < 0.05
        crypto_sig = crypto_p < 0.05

        if stock_sig:
            stock_sig_count += 1
        if crypto_sig:
            crypto_sig_count += 1

        stock_mean = result['stocks']['summary']['tarv'].get('bab_mean', np.nan) * 100
        crypto_mean = result['crypto']['summary']['tarv'].get('bab_mean', np.nan) * 100

        print(f"{freq}:")
        print(f"  Stocks:  Mean: {stock_mean:>6.3f}% | p-value: {stock_p:.4f} | "
              f"{'Significant ✓' if stock_sig else 'Not significant'}")
        print(f"  Crypto:  Mean: {crypto_mean:>6.3f}% | p-value: {crypto_p:.4f} | "
              f"{'Significant ✓' if crypto_sig else 'Not significant'}")
        print()

    print(f"Overall:")
    print(f"  Stocks: {stock_sig_count}/{len(all_results)} frequencies significant")
    print(f"  Crypto: {crypto_sig_count}/{len(all_results)} frequencies significant")

    print("\n" + "=" * 80)
    print("🎯 OVERALL CONCLUSIONS")
    print("=" * 80)

    # Interpretation 1: Stocks
    print("\n1. EQUITIES (S&P500):")
    if stock_sig_count < len(all_results) * 0.5:
        print("   ✅ Weak or inconsistent anomaly under TARV")
        print("   → Supports hypothesis: Better volatility measurement reduces anomaly")
        print("   → Suggests noise contamination in standard volatility estimates")
        print("   → Markets may be more efficient than standard measures suggest")
    else:
        print("   ⚠️  Anomaly persists even with TARV")
        print("   → Structural mispricing or risk-based explanation")
        print("   → Not purely a measurement issue")

    # Interpretation 2: Crypto
    print("\n2. CRYPTOCURRENCY:")
    if crypto_sig_count >= len(all_results) * 0.5:
        print("   ⚠️  Strong persistent anomaly under TARV")
        print("   → Reflects microstructure inefficiency")
        print("   → Behavioral dynamics dominate")
        print("   → Market less efficient than equities")
        print("   → High-frequency noise not the main driver")
    else:
        print("   ✅ Anomaly reduced or inconsistent")
        print("   → TARV successfully filters noise")
        print("   → Improving market efficiency")

    # Cross-market comparison
    print("\n3. CROSS-MARKET INSIGHTS:")
    if crypto_sig_count > stock_sig_count:
        print("   📊 Crypto shows STRONGER anomaly than stocks")
        print("   → Fundamental market structure differences")
        print("   → Less efficient price discovery in crypto")
        print("   → Greater role of behavioral factors")
    elif stock_sig_count > crypto_sig_count:
        print("   📊 Stocks show STRONGER anomaly than crypto")
        print("   → Unexpected result - requires investigation")
        print("   → May reflect different sample periods or leverage constraints")
    else:
        print("   📊 Similar anomaly strength across markets")
        print("   → Universal phenomenon")
        print("   → Not market-structure specific")

    print("\n" + "=" * 80)


def get_significance_symbol(p_value: float) -> str:
    """Get significance symbol based on p-value."""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return ""


def save_results(all_results: List[Dict], filepath: Path):
    """Save results to pickle file."""
    import pickle

    with open(filepath, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n💾 Results saved to: {filepath}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run the full cross-market comparison analysis."""
    print("\n" + "=" * 80)
    print("🌍 CROSS-MARKET COMPARISON: EQUITIES VS. CRYPTOCURRENCY")
    print("=" * 80)
    print("\nMotivation:")
    print("1. Equities and crypto differ fundamentally in:")
    print("   • Market microstructure")
    print("   • Tail behavior")
    print("   • Liquidity")
    print("   • Efficiency")
    print("2. TARV reveals how these structural differences affect volatility-return patterns")
    print("\nFrequencies: " + ", ".join(FREQUENCIES))
    print("=" * 80)

    # Configuration
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    config = {
        'crypto_exchange': 'binance',
        'start_date': '2024-11-27',
        'end_date': '2025-11-26',
        'n_portfolios': 3,
        'use_universal_alpha': True  # Use cross-market alpha estimation
    }

    # Run analysis for each frequency
    all_results = []

    for frequency in FREQUENCIES:
        try:
            result = run_cross_market_comparison(
                data_dir=data_dir,
                frequency=frequency,
                crypto_exchange=config['crypto_exchange'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                n_portfolios=config['n_portfolios'],
                use_universal_alpha=config['use_universal_alpha']
            )

            all_results.append(result)
            print(f"\n✅ Completed cross-market comparison for {frequency}")

        except Exception as e:
            print(f"\n❌ Failed to analyze {frequency}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comprehensive comparison
    if all_results:
        print_cross_market_summary(all_results)
        print_interpretation(all_results)

        # Save results
        save_results(all_results, script_dir / "cross_market_comparison_results.pkl")
    else:
        print("\n❌ No results to compare")

    return all_results


if __name__ == "__main__":
    results = main()
