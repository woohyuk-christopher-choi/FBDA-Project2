#!/usr/bin/env python3
"""
S&P500 Stock Analysis: Low-Beta Anomaly Test

Same methodology as crypto analysis but for US stocks.
Includes:
1. CAPM Test (Fama-MacBeth)
2. Low-Beta Anomaly Test
3. BAB Strategy Backtest
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

from stock_data_loader import load_sp500_data, resample_to_frequency
from portfolio import BetaEstimator, PortfolioConstructor, BacktestEngine, print_performance_summary


def run_fama_macbeth_stock(
    assets_returns: pd.DataFrame,
    market_returns: pd.Series,
    betas_by_period: Dict,
    holding_returns: Dict
) -> Dict:
    """Run Fama-MacBeth regression for stocks."""
    gamma0_list = []
    gamma1_list = []
    r2_list = []
    market_ret_list = []
    
    for period, betas in betas_by_period.items():
        if period not in holding_returns:
            continue
        
        period_returns = holding_returns[period]
        common_assets = [a for a in betas.keys() if a in period_returns.index]
        
        if len(common_assets) < 5:
            continue
        
        y = np.array([period_returns[a] for a in common_assets])
        x = np.array([betas[a] for a in common_assets])
        
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 5:
            continue
        
        y, x = y[valid], x[valid]
        
        # Cross-sectional regression
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            gamma0, gamma1 = coeffs
            
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            gamma0_list.append(gamma0)
            gamma1_list.append(gamma1)
            r2_list.append(r2)
            
            if period in market_returns.index:
                market_ret_list.append(market_returns.loc[period])
        except:
            continue
    
    if len(gamma1_list) < 10:
        return {'gamma1_mean': np.nan, 'gamma1_tstat': np.nan, 'market_premium': np.nan}
    
    gamma1_arr = np.array(gamma1_list)
    gamma1_mean = np.mean(gamma1_arr)
    gamma1_se = np.std(gamma1_arr, ddof=1) / np.sqrt(len(gamma1_arr))
    gamma1_tstat = gamma1_mean / gamma1_se if gamma1_se > 0 else 0
    
    return {
        'gamma0_mean': np.mean(gamma0_list),
        'gamma1_mean': gamma1_mean,
        'gamma1_tstat': gamma1_tstat,
        'market_premium': np.mean(market_ret_list) if market_ret_list else np.nan,
        'r2_avg': np.mean(r2_list),
        'n_periods': len(gamma1_list)
    }


def run_stock_analysis(
    stock_prices: pd.DataFrame,
    market_prices: pd.Series,
    frequency: str = '5min',
    use_tarv: bool = True,
    method_name: str = "TARV"
) -> Tuple[Dict, Dict]:
    """
    Run complete analysis for stocks.
    
    Returns:
        Tuple of (backtest_result, fama_macbeth_result)
    """
    # Resample if needed
    if frequency != '1min':
        stock_prices = resample_to_frequency(stock_prices, frequency)
        market_prices = resample_to_frequency(pd.DataFrame({'market': market_prices}), frequency)['market']
    
    # Calculate observations per day (US market: ~390 min/day = 6.5 hours)
    freq_map = {'1min': 1, '5min': 5, '15min': 15, '30min': 30, '1h': 60, '1d': 390}
    freq_minutes = freq_map.get(frequency, 1)
    
    if frequency == '1d':
        obs_per_day = 1  # 1 observation per day for daily data
    else:
        obs_per_day = 390 // freq_minutes  # US market hours (390 min/day)
    
    # Window days adjusted by frequency (same logic as crypto)
    if frequency == '1min':
        window_days = 5  # Too much data for 1min
    elif frequency in ['5min', '15min']:
        window_days = 20
    elif frequency == '30min':
        window_days = 30
    elif frequency == '1h':
        window_days = 60
    else:  # 1d
        window_days = 120  # ~6 months of daily data
    
    window_obs = obs_per_day * window_days
    min_obs = int(window_obs * 0.3)  # 30% of window
    
    print(f"\n{'='*70}")
    print(f"📊 METHOD: {method_name}")
    print(f"   Frequency: {frequency}, Window: {window_days} days ({window_obs} obs)")
    print(f"{'='*70}")
    
    # Calculate returns
    stock_returns = stock_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    
    # Align
    common_idx = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    print(f"   Data: {len(stock_returns)} observations, {len(stock_returns.columns)} stocks")
    
    # Initialize estimator
    beta_estimator = BetaEstimator(
        window_observations=window_obs,
        min_observations=min_obs,
        use_tarv=use_tarv,
        c_omega=0.333,
        use_rolling_alpha=True
    )
    
    portfolio_constructor = PortfolioConstructor(
        n_portfolios=5,
        weighting='equal',
        long_short=True
    )
    
    backtest_engine = BacktestEngine(
        beta_estimator=beta_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency='weekly',
        transaction_cost=0.001
    )
    
    # Run backtest
    result = backtest_engine.run_backtest(
        assets_prices=stock_prices.copy(),
        market_prices=market_prices.copy()
    )
    
    # Collect betas for Fama-MacBeth
    betas_by_period = {}
    holding_returns = {}
    
    rebalance_dates = pd.date_range(
        start=stock_returns.index[0],
        end=stock_returns.index[-1],
        freq='W-MON'
    )
    
    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]
        
        rebal_idx = stock_returns.index.searchsorted(rebal_date)
        if rebal_idx >= len(stock_returns):
            continue
        
        start_idx = max(0, rebal_idx - window_obs + 1)
        est_assets = stock_returns.iloc[start_idx:rebal_idx + 1]
        est_market = market_returns.iloc[start_idx:rebal_idx + 1]
        
        if len(est_assets) < min_obs:
            continue
        
        try:
            assets_dict = {col: est_assets[col].values for col in est_assets.columns}
            beta_results = beta_estimator.estimate_all_betas(assets_dict, est_market.values)
            betas = {k: v[0] for k, v in beta_results.items() if not np.isnan(v[0])}
            
            if len(betas) >= 5:
                betas_by_period[rebal_date] = betas
                
                # Holding period returns
                hold_mask = (stock_returns.index > rebal_date) & (stock_returns.index <= next_rebal)
                if hold_mask.sum() > 0:
                    period_return = (1 + stock_returns.loc[hold_mask]).prod() - 1
                    holding_returns[rebal_date] = period_return
        except:
            continue
    
    # Fama-MacBeth
    market_period_returns = {}
    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]
        hold_mask = (market_returns.index > rebal_date) & (market_returns.index <= next_rebal)
        if hold_mask.sum() > 0:
            market_period_returns[rebal_date] = (1 + market_returns.loc[hold_mask]).prod() - 1
    
    fm_result = run_fama_macbeth_stock(
        stock_returns, 
        pd.Series(market_period_returns),
        betas_by_period, 
        holding_returns
    )
    
    return result, fm_result


def main(fast_mode: bool = True):
    """Main analysis function.
    
    Args:
        fast_mode: If True, skip 1min frequency (very slow) and use fewer methods
    """
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / "data" / "S&P500_index포함_분봉_20241126_20251126.xlsx"
    
    print("\n" + "=" * 70)
    print("🔬 S&P500 STOCK ANALYSIS: LOW-BETA ANOMALY TEST")
    if fast_mode:
        print("   ⚡ FAST MODE: Skipping 1min frequency")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    stock_prices, market_prices = load_sp500_data(str(data_file))
    
    # Test different frequencies (same as crypto analysis)
    # Note: 1h = 60min, but stock market has only ~6.5 hours/day
    if fast_mode:
        frequencies = ['5min', '15min', '30min', '1h', '1d']  # Skip 1min (too slow)
    else:
        frequencies = ['1min', '5min', '15min', '30min', '1h', '1d']
    
    methods = [
        ('TARV', True),
        ('Standard RV', False)
    ]
    
    all_results = {}
    fm_results = {}
    
    total_runs = len(frequencies) * len(methods)
    current_run = 0
    
    for freq in frequencies:
        print(f"\n{'='*70}")
        print(f"📈 FREQUENCY: {freq}")
        print(f"{'='*70}")
        
        for method_name, use_tarv in methods:
            current_run += 1
            full_name = f"{method_name} ({freq})"
            print(f"\n[{current_run}/{total_runs}] {full_name}")
            
            try:
                result, fm = run_stock_analysis(
                    stock_prices.copy(),
                    market_prices.copy(),
                    frequency=freq,
                    use_tarv=use_tarv,
                    method_name=full_name
                )
                
                all_results[full_name] = result
                fm_results[full_name] = fm
                
                # Print summary
                if 'BAB' in result and 'return' in result['BAB']:
                    bab_ret = (1 + result['BAB']['return']).prod() - 1
                    print(f"   BAB Total Return: {bab_ret*100:.2f}%")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Print CAPM test results
    print("\n" + "=" * 70)
    print("📊 FAMA-MACBETH CAPM TEST RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<25} {'γ₁ (Risk Prem)':<15} {'t-stat':<10} {'Mkt Prem':<12} {'CAPM?':<10}")
    print("-" * 70)
    
    for method, fm in fm_results.items():
        if fm is None or np.isnan(fm.get('gamma1_mean', np.nan)):
            continue
        
        gamma1 = f"{fm['gamma1_mean']*100:.3f}%"
        tstat = f"{fm['gamma1_tstat']:.2f}"
        mkt = f"{fm['market_premium']*100:.3f}%" if not np.isnan(fm.get('market_premium', np.nan)) else "N/A"
        
        # Check if CAPM holds
        if not np.isnan(fm['gamma1_mean']) and not np.isnan(fm.get('market_premium', np.nan)):
            ratio = fm['gamma1_mean'] / fm['market_premium'] if fm['market_premium'] != 0 else 0
            capm = "✅" if 0.5 < ratio < 2.0 else "❌"
        else:
            capm = "N/A"
        
        print(f"{method:<25} {gamma1:<15} {tstat:<10} {mkt:<12} {capm:<10}")
    
    # Print BAB results
    print("\n" + "=" * 70)
    print("📊 BAB STRATEGY RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<25} {'BAB Return':<12} {'Sharpe':<10} {'Max DD':<12} {'p-value':<10}")
    print("-" * 70)
    
    for method, result in all_results.items():
        if 'BAB' not in result or 'return' not in result['BAB']:
            continue
        
        bab = result['BAB']
        total_ret = (1 + bab['return']).prod() - 1
        
        returns = bab['return'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
        
        cumulative = (1 + bab['return']).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        
        # t-test
        t_stat, p_val = stats.ttest_1samp(returns, 0)
        p_val = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        
        sig = "✅" if p_val < 0.05 else "❌"
        
        print(f"{method:<25} {total_ret*100:>10.2f}% {sharpe:>9.2f} {max_dd*100:>10.2f}% {p_val:>8.4f} {sig}")
    
    # Print Portfolio Returns by Beta Group
    print("\n" + "=" * 70)
    print("📊 PORTFOLIO RETURNS BY BETA GROUP (Frequency Comparison)")
    print("=" * 70)
    
    # Portfolio names for 5 portfolios: Low_Beta, Q2, Q3, Q4, High_Beta
    port_names = ['Low_Beta', 'Q2', 'Q3', 'Q4', 'High_Beta']
    
    # Organize results by method type
    tarv_results = {k: v for k, v in all_results.items() if 'TARV' in k}
    std_results = {k: v for k, v in all_results.items() if 'Standard' in k}
    
    for method_type, results_dict in [('TARV', tarv_results), ('Standard RV', std_results)]:
        print(f"\n📈 {method_type} Method:")
        print("-" * 100)
        print(f"{'Frequency':<10} {'Low_Beta':<12} {'Q2':<12} {'Q3':<12} {'Q4':<12} {'High_Beta':<12} {'BAB':<12} {'Anomaly?':<10}")
        print("-" * 100)
        
        for method_name in sorted(results_dict.keys(), key=lambda x: ['1min', '5min', '15min', '30min', '1h', '1d'].index(x.split('(')[1].replace(')', '')) if '(' in x else 0):
            result = results_dict[method_name]
            
            # Extract frequency from method name
            freq = method_name.split('(')[1].replace(')', '') if '(' in method_name else method_name
            
            row_values = []
            
            # Get portfolio returns (Low_Beta, Q2, Q3, Q4, High_Beta)
            for port_name in port_names:
                if port_name in result and 'return' in result[port_name]:
                    total_ret = (1 + result[port_name]['return']).prod() - 1
                    row_values.append(f"{total_ret*100:>+.1f}%")
                else:
                    row_values.append("N/A")
            
            # Get BAB return
            if 'BAB' in result and 'return' in result['BAB']:
                bab_ret = (1 + result['BAB']['return']).prod() - 1
                row_values.append(f"{bab_ret*100:>+.1f}%")
            else:
                row_values.append("N/A")
            
            # Check Low-Beta Anomaly (Low_Beta > High_Beta)
            low_ret = (1 + result['Low_Beta']['return']).prod() - 1 if 'Low_Beta' in result else np.nan
            high_ret = (1 + result['High_Beta']['return']).prod() - 1 if 'High_Beta' in result else np.nan
            
            if not np.isnan(low_ret) and not np.isnan(high_ret):
                anomaly = "✅ Yes" if low_ret > high_ret else "❌ No"
            else:
                anomaly = "N/A"
            row_values.append(anomaly)
            
            print(f"{freq:<10} {row_values[0]:<12} {row_values[1]:<12} {row_values[2]:<12} {row_values[3]:<12} {row_values[4]:<12} {row_values[5]:<12} {row_values[6]:<10}")
        
        print("-" * 100)
    
    # Summary: Low-Beta Anomaly check
    print("\n📋 Low-Beta Anomaly Summary:")
    print("   (Anomaly exists if Low_Beta return > High_Beta return)")
    
    anomaly_count = 0
    total_count = 0
    for result in all_results.values():
        if 'Low_Beta' in result and 'High_Beta' in result:
            low_ret = (1 + result['Low_Beta']['return']).prod() - 1
            high_ret = (1 + result['High_Beta']['return']).prod() - 1
            total_count += 1
            if low_ret > high_ret:
                anomaly_count += 1
    
    print(f"   → Anomaly detected in {anomaly_count}/{total_count} cases")
    if anomaly_count > total_count / 2:
        print("   → 📈 LOW-BETA ANOMALY EXISTS in S&P500")
    else:
        print("   → 📉 NO Low-Beta Anomaly (High-beta outperforms)")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("📋 CONCLUSION")
    print("=" * 70)
    
    # Analyze CAPM relationship (γ₁ / Market Premium ratio)
    gamma1_ratios = []
    for fm in fm_results.values():
        if fm and not np.isnan(fm.get('gamma1_mean', np.nan)):
            if not np.isnan(fm.get('market_premium', np.nan)) and fm['market_premium'] != 0:
                ratio = fm['gamma1_mean'] / fm['market_premium']
                gamma1_ratios.append(ratio)
    
    avg_ratio = np.mean(gamma1_ratios) if gamma1_ratios else np.nan
    
    print(f"\n📈 Risk-Return Relationship Analysis")
    print(f"   γ₁ / Market Premium ratio: {avg_ratio:.2f}")
    print()
    
    if np.isnan(avg_ratio):
        print("   ⚠️ Unable to determine relationship")
    elif 0.7 < avg_ratio < 1.5:
        print("   ✅ CAPM HOLDS (High-Risk = High-Return)")
        print("   → Risk-Return relationship matches CAPM prediction")
        print("   → NO Low-Beta Anomaly")
    elif avg_ratio <= 0.7:
        print("   ❌ LOW-BETA ANOMALY EXISTS")
        print(f"   → γ₁ < Market Premium (ratio = {avg_ratio:.2f})")
        print("   → High-beta assets are UNDERCOMPENSATED for risk")
        print("   → Low-beta assets outperform on risk-adjusted basis")
    else:  # avg_ratio >= 1.5
        print("   ⚠️ STRONGER THAN CAPM (High-Risk = Even Higher Return)")
        print(f"   → γ₁ > Market Premium (ratio = {avg_ratio:.2f})")
        print("   → High-beta assets are OVERCOMPENSATED for risk")
        print("   → This is OPPOSITE of Low-Beta Anomaly")
        print("   → Consistent with 'High-Risk, High-Return' but STRONGER")
    
    # Check BAB strategy
    print(f"\n📊 BAB Strategy Implication")
    
    bab_positive = 0
    bab_negative = 0
    
    for method, result in all_results.items():
        if 'BAB' in result and 'return' in result['BAB']:
            total_ret = (1 + result['BAB']['return']).prod() - 1
            if total_ret > 0:
                bab_positive += 1
            else:
                bab_negative += 1
    
    if bab_positive > bab_negative:
        print("   → BAB (Long Low-Beta, Short High-Beta) shows positive returns")
        print("   → Suggests LOW-BETA ANOMALY")
    else:
        print("   → BAB (Long Low-Beta, Short High-Beta) shows negative returns")
        print("   → High-beta outperforms → NO Low-Beta Anomaly")
        print("   → Consistent with 'High-Risk, High-Return' theory")
    
    # Final summary
    print(f"\n{'='*70}")
    print("📋 FINAL SUMMARY: S&P500 Stock Market")
    print("=" * 70)
    
    if avg_ratio <= 0.7:
        print("   🎯 LOW-BETA ANOMALY EXISTS")
        print("   → Low-beta stocks outperform high-beta stocks")
        print("   → CAPM does NOT hold")
    elif avg_ratio >= 1.5:
        print("   🎯 NO LOW-BETA ANOMALY")
        print("   → High-beta stocks outperform (even more than CAPM predicts)")
        print("   → 'High-Risk, High-Return' relationship is STRONGER than CAPM")
    else:
        print("   🎯 CAPM HOLDS (approximately)")
        print("   → Risk-Return relationship matches theory")
        print("   → No significant anomaly detected")
    
    print("=" * 70)
    
    return all_results, fm_results


if __name__ == "__main__":
    import sys
    # Use fast_mode=False to include 1min frequency
    fast_mode = "--full" not in sys.argv
    main(fast_mode=fast_mode)
