#!/usr/bin/env python3
"""
Frequency Comparison Analysis for Beta Estimation Quality

This script compares different data frequencies using:
1. Fama-MacBeth Cross-sectional Regression
2. Beta Prediction Accuracy (out-of-sample)
3. Portfolio Realized Beta vs Estimated Beta

Goal: Determine which frequency produces the most accurate beta estimates
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from portfolio import BetaEstimator, PortfolioConstructor, BacktestEngine
from data_loader import prepare_data_for_analysis, FREQUENCY_MINUTES


@dataclass
class FamaMacBethResult:
    """Results from Fama-MacBeth regression."""
    gamma0_mean: float      # Average intercept
    gamma0_tstat: float     # t-stat for intercept
    gamma1_mean: float      # Average slope (risk premium)
    gamma1_tstat: float     # t-stat for slope
    market_premium: float   # Actual market excess return
    n_periods: int          # Number of cross-sectional regressions
    r2_avg: float           # Average R² from cross-sectional regressions


@dataclass  
class BetaQualityMetrics:
    """Beta estimation quality metrics."""
    fama_macbeth: FamaMacBethResult
    beta_persistence: float      # Correlation between β_t and β_{t+1}
    realized_vs_estimated: float # Correlation between realized and estimated beta
    portfolio_beta_spread: float # High_beta - Low_beta realized spread


def run_fama_macbeth(
    assets_returns: pd.DataFrame,
    market_returns: pd.Series,
    betas_by_period: Dict[pd.Timestamp, Dict[str, float]],
    holding_returns: Dict[pd.Timestamp, pd.Series]
) -> FamaMacBethResult:
    """
    Run Fama-MacBeth two-stage regression.
    
    Stage 1: For each period t, regress cross-sectional returns on betas
             r_{i,t} = γ_{0,t} + γ_{1,t} × β_i + ε_{i,t}
    
    Stage 2: Test if average γ_1 equals market premium
    """
    gamma0_list = []
    gamma1_list = []
    r2_list = []
    market_returns_list = []
    
    for period, betas in betas_by_period.items():
        if period not in holding_returns:
            continue
            
        period_returns = holding_returns[period]
        
        # Get common assets
        common_assets = [a for a in betas.keys() if a in period_returns.index]
        if len(common_assets) < 5:  # Need enough assets for regression
            continue
        
        # Prepare data
        y = np.array([period_returns[a] for a in common_assets])
        x = np.array([betas[a] for a in common_assets])
        
        # Remove NaN
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 5:
            continue
            
        y = y[valid]
        x = x[valid]
        
        # Cross-sectional regression: r_i = γ_0 + γ_1 * β_i
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            gamma0, gamma1 = coeffs
            
            # R²
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            gamma0_list.append(gamma0)
            gamma1_list.append(gamma1)
            r2_list.append(r2)
            
            # Market return for this period
            if period in market_returns.index:
                market_returns_list.append(market_returns.loc[period])
                
        except np.linalg.LinAlgError:
            continue
    
    if len(gamma1_list) < 10:
        return FamaMacBethResult(
            gamma0_mean=np.nan, gamma0_tstat=np.nan,
            gamma1_mean=np.nan, gamma1_tstat=np.nan,
            market_premium=np.nan, n_periods=0, r2_avg=np.nan
        )
    
    # Stage 2: Average and t-test
    gamma0_arr = np.array(gamma0_list)
    gamma1_arr = np.array(gamma1_list)
    
    gamma0_mean = np.mean(gamma0_arr)
    gamma0_se = np.std(gamma0_arr, ddof=1) / np.sqrt(len(gamma0_arr))
    gamma0_tstat = gamma0_mean / gamma0_se if gamma0_se > 0 else 0
    
    gamma1_mean = np.mean(gamma1_arr)
    gamma1_se = np.std(gamma1_arr, ddof=1) / np.sqrt(len(gamma1_arr))
    gamma1_tstat = gamma1_mean / gamma1_se if gamma1_se > 0 else 0
    
    market_premium = np.mean(market_returns_list) if market_returns_list else np.nan
    
    return FamaMacBethResult(
        gamma0_mean=gamma0_mean,
        gamma0_tstat=gamma0_tstat,
        gamma1_mean=gamma1_mean,
        gamma1_tstat=gamma1_tstat,
        market_premium=market_premium,
        n_periods=len(gamma1_list),
        r2_avg=np.mean(r2_list)
    )


def calculate_beta_persistence(
    betas_by_period: Dict[pd.Timestamp, Dict[str, float]]
) -> float:
    """Calculate autocorrelation of beta estimates."""
    periods = sorted(betas_by_period.keys())
    
    if len(periods) < 2:
        return np.nan
    
    correlations = []
    for i in range(len(periods) - 1):
        t0 = periods[i]
        t1 = periods[i + 1]
        
        betas_t0 = betas_by_period[t0]
        betas_t1 = betas_by_period[t1]
        
        common = set(betas_t0.keys()) & set(betas_t1.keys())
        if len(common) < 5:
            continue
        
        b0 = np.array([betas_t0[a] for a in common])
        b1 = np.array([betas_t1[a] for a in common])
        
        valid = ~(np.isnan(b0) | np.isnan(b1))
        if valid.sum() < 5:
            continue
        
        corr = np.corrcoef(b0[valid], b1[valid])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else np.nan


def calculate_realized_beta(
    assets_returns: pd.DataFrame,
    market_returns: pd.Series,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp
) -> Dict[str, float]:
    """Calculate realized beta during holding period using OLS."""
    mask = (assets_returns.index > period_start) & (assets_returns.index <= period_end)
    
    realized_betas = {}
    for asset in assets_returns.columns:
        y = assets_returns.loc[mask, asset].values
        x = market_returns.loc[mask].values
        
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 10:
            continue
        
        y = y[valid]
        x = x[valid]
        
        # OLS beta
        cov = np.cov(y, x)[0, 1]
        var = np.var(x, ddof=1)
        if var > 0:
            realized_betas[asset] = cov / var
    
    return realized_betas


def run_frequency_analysis(
    frequency: str,
    data_dir: str,
    exchange: str = 'binance',
    start_date: str = '2023-11-27',
    end_date: str = '2025-11-26',
    use_tarv: bool = True
) -> Tuple[BetaQualityMetrics, Dict]:
    """Run complete analysis for a single frequency."""
    
    # Auto-adjust parameters based on frequency
    freq_minutes = FREQUENCY_MINUTES.get(frequency, 1)
    obs_per_day = 1440 // freq_minutes
    
    if frequency == '1d':
        window_days = 180
        min_obs = 60
    elif frequency == '1h':
        window_days = 60
        min_obs = 480
    elif frequency in ['15m', '30m']:
        window_days = 30
        min_obs = int(obs_per_day * 30 * 0.25)
    else:
        window_days = 30
        min_obs = int(obs_per_day * 30 * 0.2)
    
    # Load data
    try:
        assets_prices, market_prices = prepare_data_for_analysis(
            data_dir=data_dir,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            min_observations=min_obs,
            frequency=frequency
        )
    except Exception as e:
        print(f"  ❌ Error loading {frequency} data: {e}")
        return None, {}
    
    if assets_prices.empty:
        print(f"  ❌ No data for {frequency}")
        return None, {}
    
    # Calculate returns
    assets_returns = assets_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    
    # Align
    common_idx = assets_returns.index.intersection(market_returns.index)
    assets_returns = assets_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    # Initialize beta estimator
    beta_estimator = BetaEstimator(
        window_observations=obs_per_day * window_days,
        min_observations=min_obs,
        use_tarv=use_tarv,
        c_omega=0.333,
        use_rolling_alpha=True
    )
    
    # Generate rebalance dates (weekly)
    rebalance_dates = pd.date_range(
        start=assets_returns.index[0],
        end=assets_returns.index[-1],
        freq='W-MON'
    )
    rebalance_dates = [d for d in rebalance_dates if d in assets_returns.index or 
                      assets_returns.index.searchsorted(d) < len(assets_returns)]
    
    # Collect betas and returns by period
    betas_by_period = {}
    holding_returns = {}
    realized_betas_by_period = {}
    
    print(f"  📊 Processing {len(rebalance_dates)-1} rebalance periods...")
    
    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]
        
        # Find index position
        rebal_idx = assets_returns.index.searchsorted(rebal_date)
        if rebal_idx >= len(assets_returns):
            continue
        
        # Estimation window
        window_size = beta_estimator.window_observations
        start_idx = max(0, rebal_idx - window_size + 1)
        
        est_assets = assets_returns.iloc[start_idx:rebal_idx + 1]
        est_market = market_returns.iloc[start_idx:rebal_idx + 1]
        
        if len(est_assets) < beta_estimator.min_observations:
            continue
        
        # Estimate betas
        try:
            assets_dict = {col: est_assets[col].values for col in est_assets.columns}
            beta_results = beta_estimator.estimate_all_betas(assets_dict, est_market.values)
            betas = {k: v[0] for k, v in beta_results.items()}
            
            if not betas or all(np.isnan(list(betas.values()))):
                continue
            
            betas_by_period[rebal_date] = betas
            
            # Holding period returns (for Fama-MacBeth)
            hold_mask = (assets_returns.index > rebal_date) & (assets_returns.index <= next_rebal)
            if hold_mask.sum() > 0:
                period_return = (1 + assets_returns.loc[hold_mask]).prod() - 1
                holding_returns[rebal_date] = period_return
                
                # Realized beta during holding period
                realized_betas_by_period[rebal_date] = calculate_realized_beta(
                    assets_returns, market_returns, rebal_date, next_rebal
                )
        except Exception as e:
            continue
    
    if len(betas_by_period) < 10:
        print(f"  ❌ Not enough valid periods ({len(betas_by_period)})")
        return None, {}
    
    # Calculate market returns for Fama-MacBeth
    market_period_returns = {}
    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]
        hold_mask = (market_returns.index > rebal_date) & (market_returns.index <= next_rebal)
        if hold_mask.sum() > 0:
            market_period_returns[rebal_date] = (1 + market_returns.loc[hold_mask]).prod() - 1
    
    market_period_series = pd.Series(market_period_returns)
    
    # 1. Fama-MacBeth Test
    fm_result = run_fama_macbeth(
        assets_returns, market_period_series, betas_by_period, holding_returns
    )
    
    # 2. Beta Persistence
    persistence = calculate_beta_persistence(betas_by_period)
    
    # 3. Realized vs Estimated Beta correlation
    est_vs_real_corrs = []
    for period, est_betas in betas_by_period.items():
        if period in realized_betas_by_period:
            real_betas = realized_betas_by_period[period]
            common = set(est_betas.keys()) & set(real_betas.keys())
            if len(common) >= 5:
                est = np.array([est_betas[a] for a in common])
                real = np.array([real_betas[a] for a in common])
                valid = ~(np.isnan(est) | np.isnan(real))
                if valid.sum() >= 5:
                    corr = np.corrcoef(est[valid], real[valid])[0, 1]
                    if not np.isnan(corr):
                        est_vs_real_corrs.append(corr)
    
    realized_vs_estimated = np.mean(est_vs_real_corrs) if est_vs_real_corrs else np.nan
    
    # 4. Portfolio beta spread (High - Low)
    # Sort by beta each period and check if high-beta assets have higher realized beta
    spreads = []
    for period, est_betas in betas_by_period.items():
        if period not in realized_betas_by_period:
            continue
        
        real_betas = realized_betas_by_period[period]
        common = [a for a in est_betas.keys() if a in real_betas and not np.isnan(est_betas[a])]
        
        if len(common) < 6:
            continue
        
        # Sort by estimated beta
        sorted_assets = sorted(common, key=lambda a: est_betas[a])
        n = len(sorted_assets)
        low_assets = sorted_assets[:n//3]
        high_assets = sorted_assets[-n//3:]
        
        low_real = np.mean([real_betas[a] for a in low_assets if a in real_betas])
        high_real = np.mean([real_betas[a] for a in high_assets if a in real_betas])
        
        if not np.isnan(low_real) and not np.isnan(high_real):
            spreads.append(high_real - low_real)
    
    portfolio_beta_spread = np.mean(spreads) if spreads else np.nan
    
    metrics = BetaQualityMetrics(
        fama_macbeth=fm_result,
        beta_persistence=persistence,
        realized_vs_estimated=realized_vs_estimated,
        portfolio_beta_spread=portfolio_beta_spread
    )
    
    return metrics, {
        'betas_by_period': betas_by_period,
        'n_periods': len(betas_by_period),
        'n_assets': len(assets_prices.columns)
    }


def print_comparison_results(results: Dict[str, BetaQualityMetrics]):
    """Print comparison results across frequencies."""
    
    print("\n" + "=" * 80)
    print("📊 FREQUENCY COMPARISON: BETA ESTIMATION QUALITY")
    print("=" * 80)
    
    # Header
    print(f"\n{'Frequency':<12} {'γ₁ (Risk Prem)':<16} {'γ₁ t-stat':<12} {'Mkt Premium':<12} {'γ₁ ≈ Mkt?':<10}")
    print("-" * 80)
    
    for freq, metrics in results.items():
        if metrics is None:
            print(f"{freq:<12} {'N/A':<16}")
            continue
        
        fm = metrics.fama_macbeth
        gamma1_pct = f"{fm.gamma1_mean*100:.3f}%" if not np.isnan(fm.gamma1_mean) else "N/A"
        tstat = f"{fm.gamma1_tstat:.2f}" if not np.isnan(fm.gamma1_tstat) else "N/A"
        mkt_prem = f"{fm.market_premium*100:.3f}%" if not np.isnan(fm.market_premium) else "N/A"
        
        # Check if γ₁ ≈ market premium (good beta estimation)
        if not np.isnan(fm.gamma1_mean) and not np.isnan(fm.market_premium):
            ratio = fm.gamma1_mean / fm.market_premium if fm.market_premium != 0 else 0
            if 0.5 < ratio < 2.0:
                match = "✅ YES"
            else:
                match = "❌ NO"
        else:
            match = "N/A"
        
        print(f"{freq:<12} {gamma1_pct:<16} {tstat:<12} {mkt_prem:<12} {match:<10}")
    
    # Additional metrics
    print("\n" + "-" * 80)
    print("ADDITIONAL BETA QUALITY METRICS")
    print("-" * 80)
    print(f"{'Frequency':<12} {'β Persistence':<15} {'Est vs Real':<15} {'Port β Spread':<15} {'Avg R²':<10}")
    print("-" * 80)
    
    for freq, metrics in results.items():
        if metrics is None:
            print(f"{freq:<12} {'N/A':<15}")
            continue
        
        persist = f"{metrics.beta_persistence:.3f}" if not np.isnan(metrics.beta_persistence) else "N/A"
        est_real = f"{metrics.realized_vs_estimated:.3f}" if not np.isnan(metrics.realized_vs_estimated) else "N/A"
        spread = f"{metrics.portfolio_beta_spread:.3f}" if not np.isnan(metrics.portfolio_beta_spread) else "N/A"
        r2 = f"{metrics.fama_macbeth.r2_avg:.3f}" if not np.isnan(metrics.fama_macbeth.r2_avg) else "N/A"
        
        print(f"{freq:<12} {persist:<15} {est_real:<15} {spread:<15} {r2:<10}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
    γ₁ (Risk Premium): Should equal market excess return if beta is correctly estimated
    γ₁ t-stat:         Should be significant (|t| > 2) if beta explains returns
    β Persistence:     Higher = more stable beta estimates (0.8+ is good)
    Est vs Real:       Higher = estimated beta predicts realized beta (0.5+ is good)
    Port β Spread:     Should be positive (high-β assets have higher realized β)
    Avg R²:            Higher = beta explains more cross-sectional return variation
    """)
    
    # Find best frequency
    best_freq = None
    best_score = -np.inf
    
    for freq, metrics in results.items():
        if metrics is None:
            continue
        
        fm = metrics.fama_macbeth
        if np.isnan(fm.gamma1_tstat):
            continue
        
        # Score based on: |t-stat|, persistence, est_vs_real, R²
        score = 0
        if not np.isnan(fm.gamma1_tstat):
            score += min(abs(fm.gamma1_tstat), 5)  # Cap at 5
        if not np.isnan(metrics.beta_persistence):
            score += metrics.beta_persistence * 3
        if not np.isnan(metrics.realized_vs_estimated):
            score += metrics.realized_vs_estimated * 3
        if not np.isnan(fm.r2_avg):
            score += fm.r2_avg * 2
        
        if score > best_score:
            best_score = score
            best_freq = freq
    
    if best_freq:
        print(f"\n🏆 BEST FREQUENCY FOR BETA ESTIMATION: {best_freq}")
        print(f"   (Based on Fama-MacBeth t-stat, persistence, and prediction accuracy)")
    
    print("=" * 80)


def main():
    """Main function to run frequency comparison."""
    script_dir = Path(__file__).parent
    data_dir = str(script_dir.parent / "data")
    
    # Frequencies to test
    frequencies = ['1m', '5m', '15m', '30m', '1h']
    
    print("\n" + "=" * 80)
    print("🔬 BETA ESTIMATION QUALITY ANALYSIS BY FREQUENCY")
    print("   Using Fama-MacBeth Cross-Sectional Regression")
    print("=" * 80)
    
    results = {}
    
    for freq in frequencies:
        print(f"\n📈 Processing {freq} data...")
        
        metrics, info = run_frequency_analysis(
            frequency=freq,
            data_dir=data_dir,
            use_tarv=True
        )
        
        results[freq] = metrics
        
        if metrics:
            print(f"  ✅ Completed: {info['n_periods']} periods, {info['n_assets']} assets")
    
    # Print comparison
    print_comparison_results(results)
    
    return results


if __name__ == "__main__":
    main()
