#!/usr/bin/env python3
"""
Low-Volatility Anomaly Across Time Series Analysis

This script analyzes the low-volatility anomaly across different time horizons to understand:
1. Whether volatility estimates differ drastically by time-scale
2. If daily volatility misses high-frequency information
3. Whether TARV captures noise-filtered, information-driven variation better than standard RV

Analysis:
- Computes TARV and Standard RV at multiple horizons: '1m', '5m', '15m', '30m', '1h', '1d'
- Re-runs volatility-sorted portfolio analysis at each horizon
- Compares results to determine:
  * If anomaly shrinks with TARV → standard RV is contaminated by noise/jumps
  * If anomaly persists → structural mispricing or risk-based explanation
  * Differences across horizons offer insight into market efficiency levels
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from portfolio import BetaEstimator, PortfolioConstructor, BacktestEngine
from data_loader import prepare_data_for_analysis, FREQUENCY_MINUTES
from scipy import stats


# =============================================================================
# Configuration
# =============================================================================

FREQUENCIES = ['1m', '5m', '15m', '30m', '1h', '1d']

# Frequency-specific parameters
FREQUENCY_CONFIG = {
    '1m': {
        'window_days': 30,
        'min_obs_ratio': 0.2,
        'rebalance_frequency': 'weekly'
    },
    '5m': {
        'window_days': 30,
        'min_obs_ratio': 0.2,
        'rebalance_frequency': 'weekly'
    },
    '15m': {
        'window_days': 30,
        'min_obs_ratio': 0.25,
        'rebalance_frequency': 'weekly'
    },
    '30m': {
        'window_days': 30,
        'min_obs_ratio': 0.25,
        'rebalance_frequency': 'weekly'
    },
    '1h': {
        'window_days': 60,
        'min_obs_ratio': 0.33,
        'rebalance_frequency': 'weekly'
    },
    '1d': {
        'window_days': 180,
        'min_obs_ratio': 0.33,
        'rebalance_frequency': 'weekly'
    }
}


# =============================================================================
# Main Analysis
# =============================================================================

def run_single_frequency_analysis(
    data_dir: Path,
    exchange: str,
    frequency: str,
    start_date: str,
    end_date: str,
    n_portfolios: int = 3,
    verbose: bool = False
) -> Dict:
    """
    Run analysis for a single frequency with both TARV and Standard RV.

    Returns:
        Dict with keys: 'frequency', 'tarv_results', 'standard_results', 'metadata'
    """
    print("\n" + "=" * 80)
    print(f"🔬 ANALYZING FREQUENCY: {frequency}")
    print("=" * 80)

    # Get frequency-specific config
    freq_config = FREQUENCY_CONFIG[frequency]
    freq_minutes = FREQUENCY_MINUTES.get(frequency, 1)
    obs_per_day = 1440 // freq_minutes
    window_days = freq_config['window_days']
    min_obs = int(obs_per_day * window_days * freq_config['min_obs_ratio'])

    print(f"📊 Configuration:")
    print(f"   Observations per day: {obs_per_day}")
    print(f"   Window: {window_days} days")
    print(f"   Min observations: {min_obs}")

    # Load data
    try:
        print(f"\n📥 Loading {frequency} data...")
        assets_prices, market_prices = prepare_data_for_analysis(
            data_dir=str(data_dir),
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            min_observations=min_obs,
            frequency=frequency
        )
    except Exception as e:
        print(f"❌ Error loading data for {frequency}: {e}")
        return None

    results = {
        'frequency': frequency,
        'metadata': {
            'n_assets': len(assets_prices.columns),
            'n_observations': len(assets_prices),
            'date_range': (str(assets_prices.index[0]), str(assets_prices.index[-1])),
            'window_days': window_days,
            'obs_per_day': obs_per_day
        }
    }

    # Run with TARV (Individual Alpha)
    print("\n" + "-" * 80)
    print("📊 METHOD 1: TARV (Individual Alpha)")
    print("-" * 80)

    tarv_estimator = BetaEstimator(
        window_observations=obs_per_day * window_days,
        min_observations=min_obs,
        use_tarv=True,
        c_omega=0.333,
        use_rolling_alpha=False,
        verbose=verbose
    )

    portfolio_constructor = PortfolioConstructor(
        n_portfolios=n_portfolios,
        weighting='equal',
        long_short=True
    )

    tarv_engine = BacktestEngine(
        beta_estimator=tarv_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency=freq_config['rebalance_frequency'],
        transaction_cost=0.001
    )

    tarv_results = tarv_engine.run_backtest(
        assets_prices=assets_prices.copy(),
        market_prices=market_prices.copy()
    )
    results['tarv_results'] = tarv_results

    # Run with Standard RV
    print("\n" + "-" * 80)
    print("📊 METHOD 2: Standard RV")
    print("-" * 80)

    standard_estimator = BetaEstimator(
        window_observations=obs_per_day * window_days,
        min_observations=min_obs,
        use_tarv=False,
        verbose=verbose
    )

    standard_engine = BacktestEngine(
        beta_estimator=standard_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency=freq_config['rebalance_frequency'],
        transaction_cost=0.001
    )

    standard_results = standard_engine.run_backtest(
        assets_prices=assets_prices.copy(),
        market_prices=market_prices.copy()
    )
    results['standard_results'] = standard_results

    # Calculate summary statistics
    results['summary'] = calculate_frequency_summary(tarv_results, standard_results, frequency)

    return results


def calculate_frequency_summary(
    tarv_results: Dict[str, pd.DataFrame],
    standard_results: Dict[str, pd.DataFrame],
    frequency: str
) -> Dict:
    """Calculate summary statistics for a single frequency."""
    summary = {
        'frequency': frequency,
        'tarv': {},
        'standard': {}
    }

    # Extract key metrics for TARV
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

    if 'BAB' in tarv_results:
        bab_ret = tarv_results['BAB']['return']
        t_stat, p_val_two = stats.ttest_1samp(bab_ret, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

        summary['tarv']['bab_mean'] = bab_ret.mean()
        summary['tarv']['bab_std'] = bab_ret.std()
        summary['tarv']['bab_t_stat'] = t_stat
        summary['tarv']['bab_p_value'] = p_val
        summary['tarv']['bab_sharpe'] = bab_ret.mean() / bab_ret.std() if bab_ret.std() > 0 else 0

    # Extract key metrics for Standard RV
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

    if 'BAB' in standard_results:
        bab_ret = standard_results['BAB']['return']
        t_stat, p_val_two = stats.ttest_1samp(bab_ret, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

        summary['standard']['bab_mean'] = bab_ret.mean()
        summary['standard']['bab_std'] = bab_ret.std()
        summary['standard']['bab_t_stat'] = t_stat
        summary['standard']['bab_p_value'] = p_val
        summary['standard']['bab_sharpe'] = bab_ret.mean() / bab_ret.std() if bab_ret.std() > 0 else 0

    return summary


def main():
    """Run the full time series analysis across all frequencies."""
    print("\n" + "=" * 80)
    print("🔬 LOW-VOLATILITY ANOMALY ACROSS TIME SERIES")
    print("=" * 80)
    print("\nMotivation:")
    print("1. Volatility estimates differ drastically by time-scale")
    print("2. Daily volatility may miss high-frequency information")
    print("3. HF-based TARV captures noise-filtered, information-driven variation")
    print("\nFrequencies to analyze: " + ", ".join(FREQUENCIES))
    print("=" * 80)

    # Configuration
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    config = {
        'exchange': 'binance',
        'start_date': '2023-11-27',
        'end_date': '2025-11-26',
        'n_portfolios': 3,
    }

    # Run analysis for each frequency
    all_results = []

    for frequency in FREQUENCIES:
        try:
            freq_results = run_single_frequency_analysis(
                data_dir=data_dir,
                exchange=config['exchange'],
                frequency=frequency,
                start_date=config['start_date'],
                end_date=config['end_date'],
                n_portfolios=config['n_portfolios'],
                verbose=False
            )

            if freq_results is not None:
                all_results.append(freq_results)
                print(f"\n✅ Completed analysis for {frequency}")
        except Exception as e:
            print(f"\n❌ Failed to analyze {frequency}: {e}")
            continue

    # Generate comprehensive comparison
    if all_results:
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE COMPARISON ACROSS FREQUENCIES")
        print("=" * 80)

        print_cross_frequency_comparison(all_results)
        print_interpretation_guide(all_results)

        # Save results
        save_results(all_results, script_dir / "time_series_anomaly_results.pkl")
    else:
        print("\n❌ No results to compare")

    return all_results


# =============================================================================
# Output and Visualization
# =============================================================================

def print_cross_frequency_comparison(all_results: List[Dict]):
    """Print comparison table across all frequencies."""
    print("\n" + "-" * 80)
    print("TABLE 1: Low-High Beta Spread Across Frequencies")
    print("-" * 80)
    print(f"{'Frequency':<10} {'Method':<12} {'Spread Mean':<12} {'Std':<10} "
          f"{'t-stat':<10} {'p-value':<10} {'Sig':<8}")
    print("-" * 80)

    for result in all_results:
        freq = result['frequency']
        summary = result['summary']

        # TARV
        if 'spread_mean' in summary['tarv']:
            mean = summary['tarv']['spread_mean'] * 100
            std = summary['tarv']['spread_std'] * 100
            t = summary['tarv']['t_stat']
            p = summary['tarv']['p_value']
            sig = get_significance_symbol(p)
            print(f"{freq:<10} {'TARV':<12} {mean:>11.3f}% {std:>9.3f}% "
                  f"{t:>9.3f} {p:>9.4f} {sig:<8}")

        # Standard RV
        if 'spread_mean' in summary['standard']:
            mean = summary['standard']['spread_mean'] * 100
            std = summary['standard']['spread_std'] * 100
            t = summary['standard']['t_stat']
            p = summary['standard']['p_value']
            sig = get_significance_symbol(p)
            print(f"{'':<10} {'Standard RV':<12} {mean:>11.3f}% {std:>9.3f}% "
                  f"{t:>9.3f} {p:>9.4f} {sig:<8}")

        print()

    print("\n" + "-" * 80)
    print("TABLE 2: BAB Portfolio Performance Across Frequencies")
    print("-" * 80)
    print(f"{'Frequency':<10} {'Method':<12} {'BAB Mean':<12} {'Std':<10} "
          f"{'Sharpe':<10} {'t-stat':<10} {'p-value':<10}")
    print("-" * 80)

    for result in all_results:
        freq = result['frequency']
        summary = result['summary']

        # TARV
        if 'bab_mean' in summary['tarv']:
            mean = summary['tarv']['bab_mean'] * 100
            std = summary['tarv']['bab_std'] * 100
            sharpe = summary['tarv']['bab_sharpe']
            t = summary['tarv']['bab_t_stat']
            p = summary['tarv']['bab_p_value']
            print(f"{freq:<10} {'TARV':<12} {mean:>11.3f}% {std:>9.3f}% "
                  f"{sharpe:>9.3f} {t:>9.3f} {p:>9.4f}")

        # Standard RV
        if 'bab_mean' in summary['standard']:
            mean = summary['standard']['bab_mean'] * 100
            std = summary['standard']['bab_std'] * 100
            sharpe = summary['standard']['bab_sharpe']
            t = summary['standard']['bab_t_stat']
            p = summary['standard']['bab_p_value']
            print(f"{'':<10} {'Standard RV':<12} {mean:>11.3f}% {std:>9.3f}% "
                  f"{sharpe:>9.3f} {t:>9.3f} {p:>9.4f}")

        print()


def print_interpretation_guide(all_results: List[Dict]):
    """Print interpretation of results."""
    print("\n" + "=" * 80)
    print("📋 INTERPRETATION GUIDE")
    print("=" * 80)

    # Analysis 1: Does anomaly shrink with TARV?
    print("\n1️⃣  NOISE CONTAMINATION TEST")
    print("-" * 80)
    print("Question: Does the anomaly shrink when using TARV vs Standard RV?")
    print("Interpretation:")
    print("   • If anomaly shrinks with TARV → Standard RV contaminated by noise/jumps")
    print("   • If anomaly similar → TARV doesn't remove meaningful noise")
    print("\nResults:")

    for result in all_results:
        freq = result['frequency']
        summary = result['summary']

        if 'bab_mean' in summary['tarv'] and 'bab_mean' in summary['standard']:
            tarv_ret = summary['tarv']['bab_mean'] * 100
            std_ret = summary['standard']['bab_mean'] * 100
            diff = std_ret - tarv_ret
            pct_change = (diff / abs(std_ret) * 100) if std_ret != 0 else 0

            status = "📉 Shrinks" if diff > 0 else "📈 Increases" if diff < 0 else "➡️  Same"
            print(f"   {freq:<6} TARV: {tarv_ret:>6.3f}% | Standard: {std_ret:>6.3f}% "
                  f"| Diff: {diff:>6.3f}% ({pct_change:>5.1f}%) {status}")

    # Analysis 2: Does anomaly persist across frequencies?
    print("\n2️⃣  STRUCTURAL MISPRICING TEST")
    print("-" * 80)
    print("Question: Does the anomaly persist across different time horizons?")
    print("Interpretation:")
    print("   • If persists → Structural mispricing or risk-based explanation")
    print("   • If varies dramatically → Time-scale dependent phenomenon")
    print("\nResults (TARV method):")

    significant_freqs = []
    for result in all_results:
        freq = result['frequency']
        summary = result['summary']

        if 'bab_p_value' in summary['tarv']:
            p = summary['tarv']['bab_p_value']
            mean = summary['tarv']['bab_mean'] * 100
            sig_symbol = get_significance_symbol(p)

            print(f"   {freq:<6} Mean: {mean:>6.3f}% | p-value: {p:.4f} {sig_symbol}")

            if p < 0.05:
                significant_freqs.append(freq)

    if len(significant_freqs) >= len(all_results) * 0.5:
        print("\n   ✅ Anomaly PERSISTS across most frequencies")
        print("      → Suggests structural mispricing or risk factor")
    else:
        print("\n   ⚠️  Anomaly NOT consistent across frequencies")
        print("      → Time-scale dependent or data-specific")

    # Analysis 3: Market efficiency across horizons
    print("\n3️⃣  MARKET EFFICIENCY ANALYSIS")
    print("-" * 80)
    print("Question: How does anomaly strength vary across time horizons?")
    print("Interpretation:")
    print("   • Stronger at HF → Market inefficiency at short horizons")
    print("   • Stronger at LF → Long-term structural mispricing")
    print("   • Similar across → Time-scale invariant factor")
    print("\nResults (t-statistics for TARV):")

    t_stats = []
    for result in all_results:
        freq = result['frequency']
        summary = result['summary']

        if 'bab_t_stat' in summary['tarv']:
            t = summary['tarv']['bab_t_stat']
            t_stats.append((freq, t))
            print(f"   {freq:<6} t-stat: {t:>7.3f}")

    if t_stats:
        max_freq, max_t = max(t_stats, key=lambda x: x[1])
        min_freq, min_t = min(t_stats, key=lambda x: x[1])

        print(f"\n   Strongest at: {max_freq} (t={max_t:.3f})")
        print(f"   Weakest at:   {min_freq} (t={min_t:.3f})")

    # Overall conclusion
    print("\n" + "=" * 80)
    print("🎯 OVERALL CONCLUSION")
    print("=" * 80)

    # Count significant results
    tarv_sig_count = sum(1 for r in all_results
                         if r['summary']['tarv'].get('bab_p_value', 1) < 0.05)
    std_sig_count = sum(1 for r in all_results
                        if r['summary']['standard'].get('bab_p_value', 1) < 0.05)

    print(f"\nSignificant results (p < 0.05):")
    print(f"   TARV:        {tarv_sig_count}/{len(all_results)} frequencies")
    print(f"   Standard RV: {std_sig_count}/{len(all_results)} frequencies")

    if tarv_sig_count > std_sig_count:
        print("\n✅ TARV provides STRONGER anomaly detection")
        print("   → Noise reduction improves signal quality")
    elif std_sig_count > tarv_sig_count:
        print("\n⚠️  Standard RV shows STRONGER anomaly")
        print("   → TARV may be removing true signal (over-filtering)")
    else:
        print("\n➡️  TARV and Standard RV perform SIMILARLY")
        print("   → Noise not a major factor in this market")

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
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = main()
