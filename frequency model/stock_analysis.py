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
    freq_map = {'1min': 1, '5min': 5, '15min': 15, '30min': 30, '1h': 60}
    freq_minutes = freq_map.get(frequency, 1)
    obs_per_day = 390 // freq_minutes  # US market hours (390 min/day)
    
    # Window days adjusted by frequency (same logic as crypto)
    if frequency == '1min':
        window_days = 5  # Too much data for 1min
    elif frequency in ['5min', '15min']:
        window_days = 20
    elif frequency == '30min':
        window_days = 30
    else:  # 1h
        window_days = 60
    
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


def main():
    """Main analysis function."""
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / "data" / "S&P500_index포함_분봉_20241126_20251126.xlsx"
    
    print("\n" + "=" * 70)
    print("🔬 S&P500 STOCK ANALYSIS: LOW-BETA ANOMALY TEST")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    stock_prices, market_prices = load_sp500_data(str(data_file))
    
    # Test different frequencies (same as crypto analysis)
    # Note: 1h = 60min, but stock market has only ~6.5 hours/day
    frequencies = ['1min', '5min', '15min', '30min', '1h']
    methods = [
        ('TARV', True),
        ('Standard RV', False)
    ]
    
    all_results = {}
    fm_results = {}
    
    for freq in frequencies:
        print(f"\n{'='*70}")
        print(f"📈 FREQUENCY: {freq}")
        print(f"{'='*70}")
        
        for method_name, use_tarv in methods:
            full_name = f"{method_name} ({freq})"
            
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
    
    # Conclusion
    print("\n" + "=" * 70)
    print("📋 CONCLUSION")
    print("=" * 70)
    
    # Check if CAPM holds
    capm_holds = False
    for fm in fm_results.values():
        if fm and not np.isnan(fm.get('gamma1_mean', np.nan)):
            if not np.isnan(fm.get('market_premium', np.nan)) and fm['market_premium'] != 0:
                ratio = fm['gamma1_mean'] / fm['market_premium']
                if 0.5 < ratio < 2.0:
                    capm_holds = True
                    break
    
    if capm_holds:
        print("✅ CAPM appears to hold for S&P500 stocks")
        print("   → Beta-Return relationship is as expected")
    else:
        print("❌ CAPM does NOT hold for S&P500 stocks")
        print("   → Low-Beta Anomaly may exist")
    
    # Check BAB significance
    bab_significant = False
    for result in all_results.values():
        if 'BAB' in result and 'return' in result['BAB']:
            t_stat, p_val = stats.ttest_1samp(result['BAB']['return'].values, 0)
            if t_stat > 0 and p_val / 2 < 0.05:
                bab_significant = True
                break
    
    if bab_significant:
        print("\n✅ BAB Strategy shows significant positive returns")
        print("   → Low-Beta Anomaly EXISTS in S&P500!")
    else:
        print("\n❌ BAB Strategy is not statistically significant")
        print("   → Insufficient evidence for Low-Beta Anomaly")
    
    print("=" * 70)
    
    return all_results, fm_results


if __name__ == "__main__":
    main()
