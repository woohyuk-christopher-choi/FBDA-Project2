#!/usr/bin/env python3
"""
Low-Beta Anomaly Analysis Entry Point
Compares:
1. TARV with Individual Alpha (per-asset tail index estimation)
2. TARV with Rolling Global Alpha (window-based universal alpha, no look-ahead)
3. Standard RV (no truncation)
"""

from pathlib import Path
import numpy as np
from portfolio import BetaEstimator, PortfolioConstructor, BacktestEngine, print_performance_summary
from data_loader import prepare_data_for_analysis, FREQUENCY_MINUTES


def main():
    """Run the analysis comparing 3 methods with proper rolling estimation."""
    # Get data directory relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Common parameters
    frequency = '1h'  # '1m', '5m', '15m', '30m', '1h', '1d'
    
    # Auto-adjust parameters based on frequency
    freq_minutes = FREQUENCY_MINUTES.get(frequency, 1)
    obs_per_day = 1440 // freq_minutes
    
    # For lower frequency data, need longer window for sufficient observations
    # 1m: 30 days (43,200 obs), 5m: 30 days (8,640 obs), ..., 1d: 180 days (180 obs)
    if frequency == '1d':
        window_days = 180  # 6 months for daily data
        min_obs = 60       # ~33% of window
    elif frequency == '1h':
        window_days = 60   # 2 months for hourly
        min_obs = 480      # ~33% of window (60*24*0.33)
    elif frequency in ['15m', '30m']:
        window_days = 30
        min_obs = int(obs_per_day * 30 * 0.25)
    else:  # 1m, 5m
        window_days = 30
        min_obs = int(obs_per_day * 30 * 0.2)
    
    config = {
        'exchange': 'binance',
        'start_date': '2023-11-27',
        'end_date': '2025-11-26',
        'n_portfolios': 3,
        'rebalance_frequency': 'weekly',
        'window_days': window_days,
        'frequency': frequency,
        'min_observations': min_obs,
    }
    
    print("\n" + "=" * 70)
    print("🔬 LOW-BETA ANOMALY TEST")
    print("   (Rolling Window Alpha Estimation - No Look-Ahead Bias)") 
    print("=" * 70)
    print(f"Exchange: {config['exchange'].upper()}")
    print(f"Period: {config['start_date']} to {config['end_date']}")
    print(f"Frequency: {config['frequency']}")
    print("=" * 70)
    
    # Load data once
    print("\n📥 Loading data...")
    assets_prices, market_prices = prepare_data_for_analysis(
        data_dir=str(data_dir),
        exchange=config['exchange'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        min_observations=config['min_observations'],
        frequency=config['frequency']
    )
    
    results = {}
    
    # Define 3 methods to compare
    # use_rolling_alpha=True means alpha is estimated from each window (no look-ahead)
    methods = [
        ('TARV (Individual α)', True, False),    # Individual alpha per asset
        ('TARV (Rolling Global α)', True, True), # Rolling window global alpha
        ('Standard RV', False, False),            # No TARV
    ]
    
    for method_name, use_tarv, use_rolling_alpha in methods:
        print("\n" + "=" * 70)
        print(f"📊 METHOD: {method_name}")
        if use_rolling_alpha:
            print("   α estimated from each rolling window (no look-ahead bias)")
        print("=" * 70)
        
        beta_estimator = BetaEstimator(
            window_observations=obs_per_day * config['window_days'],  # Number of observations
            min_observations=config['min_observations'],
            use_tarv=use_tarv,
            c_omega=0.333,
            use_rolling_alpha=use_rolling_alpha
        )
        
        portfolio_constructor = PortfolioConstructor(
            n_portfolios=config['n_portfolios'],
            weighting='equal',
            long_short=True
        )
        
        backtest_engine = BacktestEngine(
            beta_estimator=beta_estimator,
            portfolio_constructor=portfolio_constructor,
            rebalance_frequency=config['rebalance_frequency'],
            transaction_cost=0.001
        )
        
        result = backtest_engine.run_backtest(
            assets_prices=assets_prices.copy(),
            market_prices=market_prices.copy()
        )
        
        print_performance_summary(result, backtest_engine)
        results[method_name] = result
    
    # Print comparison and hypothesis test
    print_comparison_summary(results)
    print_hypothesis_test(results)
    
    return results


def print_comparison_summary(results: dict):
    """Print comparison summary between all methods."""
    print("\n" + "=" * 70)
    print("📈 THREE-WAY COMPARISON SUMMARY")
    print("=" * 70)
    
    method_names = list(results.keys())
    
    # Header
    print(f"\n{'Portfolio':<15}", end="")
    for name in method_names:
        short_name = name.replace('TARV ', '').replace('(', '').replace(')', '')[:12]
        print(f"{short_name:>15}", end="")
    print()
    print("-" * (15 + 15 * len(method_names)))
    
    # Portfolio returns
    for port_name in ['Low_Beta', 'Mid_Beta', 'High_Beta', 'BAB', 'Market']:
        print(f"{port_name:<15}", end="")
        for method_name in method_names:
            if port_name in results[method_name]:
                total_ret = (1 + results[method_name][port_name]['return']).prod() - 1
                print(f"{total_ret*100:>14.2f}%", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()


def print_hypothesis_test(results: dict):
    """Print formal hypothesis test for Low-Beta Anomaly."""
    print("\n" + "=" * 70)
    print("📊 HYPOTHESIS TEST: LOW-BETA ANOMALY")
    print("=" * 70)
    print("H0: Low-Beta Anomaly does not exist (Low-High spread = 0)")
    print("H1: Low-Beta assets outperform High-Beta assets (spread > 0)")
    print("-" * 70)
    
    from scipy import stats
    
    method_names = list(results.keys())
    
    print(f"\n{'Method':<25} {'L-H Spread':>10} {'t-stat':>10} {'p-value':>10} {'Result':>12}")
    print("-" * 70)
    
    for method_name in method_names:
        data = results[method_name]
        
        if 'Low_Beta' not in data or 'High_Beta' not in data:
            continue
        
        # Calculate Low - High spread for each period
        low_ret = data['Low_Beta']['return']
        high_ret = data['High_Beta']['return']
        
        # Align indices
        common_idx = low_ret.index.intersection(high_ret.index)
        low_ret = low_ret.loc[common_idx]
        high_ret = high_ret.loc[common_idx]
        
        spread = low_ret - high_ret  # Low - High spread
        
        # One-sided t-test: H1: spread > 0
        t_stat, p_val_two = stats.ttest_1samp(spread, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2  # One-sided
        
        mean_spread = spread.mean() * 100
        
        if p_val < 0.01:
            sig = "✅✅ p<0.01"
        elif p_val < 0.05:
            sig = "✅ p<0.05"
        elif p_val < 0.10:
            sig = "⚠️ p<0.10"
        else:
            sig = "❌ NS"
        
        print(f"{method_name:<25} {mean_spread:>9.3f}% {t_stat:>10.3f} {p_val:>10.4f} {sig:>12}")
    
    # BAB test (Long Low, Short High)
    print("\n" + "-" * 70)
    print("BAB STRATEGY TEST (Beta-Adjusted Long-Short)")
    print("-" * 70)
    print(f"{'Method':<25} {'BAB Mean':>10} {'t-stat':>10} {'p-value':>10} {'Result':>12}")
    print("-" * 70)
    
    for method_name in method_names:
        data = results[method_name]
        
        if 'BAB' not in data or 'return' not in data['BAB'].columns:
            continue
        
        bab_ret = data['BAB']['return']
        t_stat, p_val_two = stats.ttest_1samp(bab_ret, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2
        
        mean_ret = bab_ret.mean() * 100
        
        if p_val < 0.01:
            sig = "✅✅ p<0.01"
        elif p_val < 0.05:
            sig = "✅ p<0.05"
        elif p_val < 0.10:
            sig = "⚠️ p<0.10"
        else:
            sig = "❌ NS"
        
        print(f"{method_name:<25} {mean_ret:>9.3f}% {t_stat:>10.3f} {p_val:>10.4f} {sig:>12}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("📋 CONCLUSION")
    print("=" * 70)
    
    # Find best method
    best_method = None
    best_t = -float('inf')
    
    for method_name in method_names:
        data = results[method_name]
        if 'Low_Beta' in data and 'High_Beta' in data:
            low_ret = data['Low_Beta']['return']
            high_ret = data['High_Beta']['return']
            common_idx = low_ret.index.intersection(high_ret.index)
            spread = low_ret.loc[common_idx] - high_ret.loc[common_idx]
            t_stat, _ = stats.ttest_1samp(spread, 0)
            if t_stat > best_t:
                best_t = t_stat
                best_method = method_name
    
    if best_method:
        data = results[best_method]
        low_ret = data['Low_Beta']['return']
        high_ret = data['High_Beta']['return']
        common_idx = low_ret.index.intersection(high_ret.index)
        spread = low_ret.loc[common_idx] - high_ret.loc[common_idx]
        _, p_val_two = stats.ttest_1samp(spread, 0)
        p_val = p_val_two / 2 if best_t > 0 else 1 - p_val_two / 2
        
        if p_val < 0.05:
            print(f"✅ LOW-BETA ANOMALY EXISTS in cryptocurrency market")
            print(f"   Best method: {best_method}")
            print(f"   Evidence: t={best_t:.3f}, p={p_val:.4f}")
        else:
            print(f"❌ INSUFFICIENT EVIDENCE for Low-Beta Anomaly")
            print(f"   Best method: {best_method}, but p={p_val:.4f} >= 0.05")
    
    print("=" * 70)


if __name__ == "__main__":
    main()