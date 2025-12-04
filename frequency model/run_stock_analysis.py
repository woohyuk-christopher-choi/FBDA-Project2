#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
S&P500 STOCK LOW-BETA ANOMALY ANALYSIS
=============================================================================

This script performs comprehensive analysis of the Low-Beta Anomaly in 
US stock markets using high-frequency data from S&P500 constituents.

Six beta estimation methods are compared:
1. TARV Individual α    : Original TARV (tail-first), per-asset alpha
2. TARV Rolling α       : Original TARV with rolling universal alpha
3. Truncate-First Ind α : Truncation before tail estimation, per-asset alpha
4. Truncate-First Roll α: Truncation before tail estimation, rolling alpha
5. Simple Truncation    : Fixed 95% quantile truncation (no tail-adaptive)
6. Standard RV          : No truncation, simple realized variance/covariance

Data Source: S&P500 constituent stocks (minute-level data)
Frequencies: 5min, 15min, 30min, 1h, 1d

Usage:
    python run_stock_analysis.py

Output:
    - stock_analysis_results.csv: Detailed results for all methods/frequencies
    - stock_analysis_summary.csv: Summary statistics
    - stock_bab_returns.csv: BAB returns table
    - Console output with comprehensive analysis

Author: FBDA Project Team
Date: 2025
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stock_data_loader import load_preprocessed_data, STOCK_FREQUENCY_MINUTES
from tarv import (
    calculate_realized_beta,
    calculate_realized_beta_truncate_first,
    calculate_realized_beta_simple_truncation,
    estimate_universal_alpha
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# US Stock Market: 390 trading minutes per day (9:30 AM - 4:00 PM)
STOCK_MINUTES_PER_DAY = 390

# Frequency mapping for US stock market
STOCK_FREQUENCY_MINUTES = {
    '1min': 1,
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1h': 60,
    '1d': 390
}

# =============================================================================
# CONFIGURATION - Following Hollstein et al. (2020) & Frazzini & Pedersen (2014)
# =============================================================================

CONFIG = {
    # Include 1min to test if TARV can handle microstructure noise
    'frequencies': ['1min', '5min', '15min', '30min', '1h', '1d'],
    
    # Hollstein et al. (2020): Optimal window for beta estimation
    # Shorter windows for HF data (sufficient obs + timely), longer for daily
    # Trading days basis (~21 days/month, 252 days/year)
    'window_days': {
        '1min': 22,    # 1 month (~8,580 obs) - Hollstein recommended
        '5min': 22,    # 1 month (~1,716 obs)
        '15min': 44,   # 2 months (~1,144 obs)
        '30min': 44,   # 2 months (~572 obs)
        '1h': 63,      # 3 months (~390 obs)
        '1d': 252      # 12 months (252 obs) - traditional
    },
    
    # Frazzini & Pedersen (2014) methodology
    'rebalancing': 'monthly',
    'beta_shrinkage': 0.6,     # β_shrunk = 0.6 × β + 0.4 × 1.0
    'n_portfolios': 5,         # Quintile portfolios (5 groups for ~500 stocks)
    
    # Minimum observations per frequency (for valid beta estimation)
    'min_observations': {
        '1min': 2000,    # ~5 days of 1min data
        '5min': 500,     # ~6 days of 5min data
        '15min': 200,    # ~8 days of 15min data
        '30min': 100,    # ~8 days of 30min data
        '1h': 50,        # ~8 days of 1h data
        '1d': 60         # ~3 months of daily data
    },
    
    'methods': {
        'TARV_Ind': 'TARV Individual α',
        'TARV_Roll': 'TARV Rolling α',
        'TruncFirst_Ind': 'Truncate-First Ind α',
        'TruncFirst_Roll': 'Truncate-First Roll α',
        'SimpleTrunc': 'Simple Truncation',
        'StandardRV': 'Standard RV'
    }
}


def calculate_realized_beta_next_period(
    asset_returns: np.ndarray,
    market_returns: np.ndarray
) -> float:
    """
    Calculate realized beta for the next period (ground truth for prediction).
    Simple OLS: β = Cov(r_i, r_m) / Var(r_m)
    """
    valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
    asset_clean = asset_returns[valid_mask]
    market_clean = market_returns[valid_mask]
    
    if len(asset_clean) < 10:
        return np.nan
    
    cov = np.cov(asset_clean, market_clean)[0, 1]
    var = np.var(market_clean, ddof=1)
    
    if var == 0:
        return np.nan
    
    return cov / var


def apply_beta_shrinkage(beta: float, shrinkage: float = 0.6) -> float:
    """
    Apply Vasicek shrinkage to beta following Frazzini & Pedersen (2014).
    
    β_shrunk = shrinkage × β_raw + (1 - shrinkage) × 1.0
    
    This shrinks extreme betas toward 1.0.
    """
    if np.isnan(beta) or np.isinf(beta):
        return np.nan
    return shrinkage * beta + (1 - shrinkage) * 1.0


def calculate_bab_return(
    low_beta_return: float,
    high_beta_return: float,
    low_beta_avg: float,
    high_beta_avg: float,
    rf: float = 0.0
) -> float:
    """
    Calculate BAB return following Frazzini & Pedersen (2014).
    
    BAB = (1/β_L) × (R_L - rf) - (1/β_H) × (R_H - rf)
    
    This leverages low-beta portfolio and deleverages high-beta portfolio
    to create a zero-beta, zero-investment strategy.
    """
    if low_beta_avg <= 0 or high_beta_avg <= 0:
        return low_beta_return - high_beta_return  # Fallback to simple BAB
    
    levered_low = (1.0 / low_beta_avg) * (low_beta_return - rf)
    levered_high = (1.0 / high_beta_avg) * (high_beta_return - rf)
    
    return levered_low - levered_high


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stock_data_for_frequency(
    data_dir: Path,
    frequency: str
) -> tuple:
    """
    Load preprocessed S&P500 data for a specific frequency.
    
    Args:
        data_dir: Path to data directory
        frequency: Target frequency ('5min', '15min', '30min', '1h', '1d')
    
    Returns:
        Tuple of (stock_prices, market_prices)
    """
    # Load preprocessed data
    stock_prices, market_prices = load_preprocessed_data(str(data_dir), frequency)
    
    return stock_prices, market_prices


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_frequency(
    frequency: str,
    data_dir: Path
) -> tuple:
    """
    Analyze all 6 methods for a single frequency.
    
    Following Hollstein et al. (2020) & Frazzini & Pedersen (2014):
    - Beta prediction accuracy (RMSE)
    - Monthly rebalancing
    - Beta shrinkage: β_shrunk = 0.6 × β + 0.4 × 1.0
    - Decile portfolios (10 groups)
    - BAB with leverage adjustment: (1/β_L) × R_L - (1/β_H) × R_H
    
    Returns:
        Tuple of (portfolio_results_df, prediction_results_df)
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Frequency: {frequency}")
    print('='*60)
    
    # Get config - Hollstein et al. (2020) window sizes
    window_days = CONFIG['window_days'].get(frequency, 22)
    n_portfolios = CONFIG['n_portfolios']
    beta_shrinkage = CONFIG['beta_shrinkage']
    
    # Load data
    print("Loading data...")
    try:
        stock_prices, market_prices = load_stock_data_for_frequency(data_dir, frequency)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate returns
    stock_returns = stock_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    
    # Align
    common_idx = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    n_stocks = len(stock_returns.columns)
    n_obs = len(stock_returns)
    
    # Calculate window sizes (US market: 390 min/day)
    freq_minutes = STOCK_FREQUENCY_MINUTES.get(frequency, 1)
    obs_per_day = STOCK_MINUTES_PER_DAY // freq_minutes
    window_obs = obs_per_day * window_days
    
    print(f"   Stocks: {n_stocks}")
    print(f"   Observations: {n_obs:,}")
    print(f"   Date Range: {stock_returns.index[0]} to {stock_returns.index[-1]}")
    print(f"   Window: {window_days} trading days ({window_obs:,} obs) - Hollstein et al. (2020)")
    print(f"   Beta Shrinkage: {beta_shrinkage} (Vasicek)")
    print(f"   Portfolios: {n_portfolios} (Quintile)")
    
    # Check minimum observations
    min_obs = CONFIG['min_observations'].get(frequency, 1000)
    if n_obs < min_obs:
        print(f"[WARNING] Not enough observations ({n_obs} < {min_obs})")
    
    # For daily data, ensure K >= 3
    if frequency == '1d':
        K_base = max(3, int(0.333 * np.sqrt(window_obs)))
        min_asset_obs = 20
    else:
        K_base = max(2, int(0.333 * np.sqrt(window_obs)))
        min_asset_obs = 50
    
    print(f"   Window: {window_days} days ({window_obs:,} obs)")
    print(f"   Rebalancing: Monthly (following paper)")
    print(f"   K (approx): {K_base}")
    
    # Get rebalance points - MONTHLY (following Frazzini & Pedersen 2014)
    rebalance_dates = pd.date_range(
        start=stock_returns.index[0],
        end=stock_returns.index[-1],
        freq='MS'  # Month Start - monthly rebalancing
    )
    
    # Results storage - Quintile portfolios (Q1=Low Beta to Q5=High Beta)
    methods = CONFIG['methods']
    port_names = ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High', 'BAB', 'Market']
    all_results = {m: {p: [] for p in port_names} for m in methods.keys()}
    
    # Store average betas for each portfolio (for leverage adjustment)
    port_betas = {m: {p: [] for p in port_names} for m in methods.keys()}
    
    # Beta prediction tracking (Hollstein et al. 2020)
    # Store (predicted_beta, realized_beta) pairs for each method
    beta_predictions = {m: {'predicted': [], 'realized': []} for m in methods.keys()}
    
    # Process each rebalance period
    n_periods = 0
    min_obs_required = max(20, int(window_obs * 0.3))
    
    print("\n[PROCESSING] Rebalance periods (Monthly)...")
    print("   Measuring beta prediction accuracy (Hollstein et al. 2020)...")
    
    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]
        
        rebal_idx = stock_returns.index.searchsorted(rebal_date)
        if rebal_idx >= len(stock_returns):
            continue
        
        start_idx = max(0, rebal_idx - window_obs + 1)
        est_returns = stock_returns.iloc[start_idx:rebal_idx + 1]
        est_market = market_returns.iloc[start_idx:rebal_idx + 1].values
        
        if len(est_returns) < min_obs_required:
            continue
        
        n_obs_period = len(est_market)
        K = max(3 if frequency == '1d' else 2, int(0.333 * np.sqrt(n_obs_period)))
        
        # Calculate universal alpha for rolling methods
        all_ret = [est_returns[col].dropna().values for col in est_returns.columns]
        all_ret.append(est_market)
        try:
            universal_alpha, _ = estimate_universal_alpha(all_ret, K=K, method='median')
        except Exception:
            universal_alpha = 2.0
        
        # Calculate betas for each method
        method_betas = {}
        
        for method_key in methods.keys():
            betas = {}
            for col in est_returns.columns:
                asset_ret = est_returns[col].values
                valid_mask = ~(np.isnan(asset_ret) | np.isnan(est_market))
                asset_clean = asset_ret[valid_mask]
                market_clean = est_market[valid_mask]
                
                if len(asset_clean) < min_asset_obs:
                    continue
                
                try:
                    if method_key == 'StandardRV':
                        cov = np.sum(asset_clean * market_clean)
                        var = np.sum(market_clean ** 2)
                        beta = cov / var if var != 0 else np.nan
                    elif method_key == 'SimpleTrunc':
                        beta, _ = calculate_realized_beta_simple_truncation(
                            asset_clean, market_clean, K=K
                        )
                    elif method_key == 'TruncFirst_Ind':
                        beta, _ = calculate_realized_beta_truncate_first(
                            asset_clean, market_clean, K=K, universal_alpha=None
                        )
                    elif method_key == 'TruncFirst_Roll':
                        beta, _ = calculate_realized_beta_truncate_first(
                            asset_clean, market_clean, K=K, universal_alpha=universal_alpha
                        )
                    elif method_key == 'TARV_Ind':
                        beta, _ = calculate_realized_beta(
                            asset_clean, market_clean, K=K, use_tarv=True, universal_alpha=None
                        )
                    elif method_key == 'TARV_Roll':
                        beta, _ = calculate_realized_beta(
                            asset_clean, market_clean, K=K, use_tarv=True, universal_alpha=universal_alpha
                        )
                    else:
                        beta = np.nan
                    
                    if not np.isnan(beta) and not np.isinf(beta):
                        betas[col] = beta
                except Exception:
                    continue
            
            method_betas[method_key] = betas
        
        # Calculate realized betas for NEXT period (ground truth for prediction)
        # This is the key addition from Hollstein et al. (2020)
        next_hold_mask = (stock_returns.index > rebal_date) & (stock_returns.index <= next_rebal)
        if next_hold_mask.sum() < 10:
            continue
            
        next_returns = stock_returns.loc[next_hold_mask]
        next_market = market_returns.loc[next_hold_mask].values
        
        # Calculate realized beta for each asset in the next period
        realized_betas_next = {}
        for col in next_returns.columns:
            next_asset = next_returns[col].values
            realized_beta = calculate_realized_beta_next_period(next_asset, next_market)
            if not np.isnan(realized_beta):
                realized_betas_next[col] = realized_beta
        
        # Store prediction pairs for RMSE calculation
        for method_key, pred_betas in method_betas.items():
            for asset, pred_beta in pred_betas.items():
                if asset in realized_betas_next:
                    beta_predictions[method_key]['predicted'].append(pred_beta)
                    beta_predictions[method_key]['realized'].append(realized_betas_next[asset])
        
        # Construct portfolios for each method
        hold_mask = (stock_returns.index > rebal_date) & (stock_returns.index <= next_rebal)
        if hold_mask.sum() == 0:
            continue
        
        hold_returns = stock_returns.loc[hold_mask]
        period_returns = (1 + hold_returns).prod() - 1
        
        hold_market = market_returns.loc[hold_mask]
        market_ret = float((1 + hold_market).prod() - 1)
        
        n_periods += 1
        
        for method_key, betas in method_betas.items():
            if len(betas) < n_portfolios:  # Need at least n_portfolios assets
                continue
            
            # Apply beta shrinkage (Vasicek): β_shrunk = 0.6 × β + 0.4 × 1
            shrunk_betas = {k: apply_beta_shrinkage(v, beta_shrinkage) for k, v in betas.items()}
            shrunk_betas = {k: v for k, v in shrunk_betas.items() if not np.isnan(v)}
            
            if len(shrunk_betas) < n_portfolios:
                continue
            
            # Sort by shrunk beta
            sorted_assets = sorted(shrunk_betas.items(), key=lambda x: x[1])
            n_assets_total = len(sorted_assets)
            per_port = n_assets_total // n_portfolios
            
            # Quintile portfolio names (5 groups)
            quintile_names = ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']
            port_returns_dict = {}
            port_avg_betas = {}
            
            for j, port_name in enumerate(quintile_names):
                start = j * per_port
                end = n_assets_total if j == 4 else (j + 1) * per_port
                port_assets = [(a, b) for a, b in sorted_assets[start:end]]
                
                if len(port_assets) == 0:
                    continue
                
                # Equal weight within portfolio
                weight = 1.0 / len(port_assets)
                port_ret = 0
                port_beta_sum = 0
                valid_count = 0
                
                for asset, beta in port_assets:
                    if asset in period_returns.index:
                        r = period_returns[asset]
                        if not np.isnan(r):
                            port_ret += weight * r
                            port_beta_sum += beta
                            valid_count += 1
                
                if valid_count > 0:
                    port_ret = port_ret * len(port_assets) / valid_count
                    avg_beta = port_beta_sum / valid_count
                else:
                    avg_beta = 1.0
                
                port_returns_dict[port_name] = port_ret
                port_avg_betas[port_name] = avg_beta
                all_results[method_key][port_name].append(port_ret)
                port_betas[method_key][port_name].append(avg_beta)
            
            # BAB with leverage adjustment (Frazzini & Pedersen 2014)
            # BAB = (1/β_L) × R_L - (1/β_H) × R_H
            low_ret = port_returns_dict.get('Q1_Low', 0)
            high_ret = port_returns_dict.get('Q5_High', 0)
            low_beta = port_avg_betas.get('Q1_Low', 1.0)
            high_beta = port_avg_betas.get('Q5_High', 1.0)
            
            bab_ret = calculate_bab_return(low_ret, high_ret, low_beta, high_beta)
            all_results[method_key]['BAB'].append(bab_ret)
            all_results[method_key]['Market'].append(market_ret)
    
    print(f"   [OK] Processed {n_periods} rebalance periods (Monthly)")
    
    # Calculate Beta Prediction RMSE (Hollstein et al. 2020)
    print("\n   [RMSE] Beta Prediction Accuracy:")
    prediction_results = []
    
    for method_key, method_name in methods.items():
        preds = np.array(beta_predictions[method_key]['predicted'])
        reals = np.array(beta_predictions[method_key]['realized'])
        
        if len(preds) > 10:
            rmse = np.sqrt(np.mean((preds - reals) ** 2))
            mae = np.mean(np.abs(preds - reals))
            corr = np.corrcoef(preds, reals)[0, 1]
            n_pairs = len(preds)
        else:
            rmse = mae = corr = np.nan
            n_pairs = len(preds)
        
        prediction_results.append({
            'Frequency': frequency,
            'Method': method_key,
            'Method_Name': method_name,
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': corr,
            'N_Predictions': n_pairs
        })
        
        if not np.isnan(rmse):
            print(f"      {method_name}: RMSE={rmse:.4f}, Corr={corr:.4f}")
    
    # Aggregate portfolio results into DataFrame
    results_list = []
    
    for method_key, method_name in methods.items():
        for pname in port_names:
            returns_list = all_results[method_key][pname]
            if returns_list:
                total_ret = (np.prod([1 + r for r in returns_list]) - 1) * 100
                avg_ret = np.mean(returns_list) * 100
                
                # Get average beta for this portfolio
                beta_list = port_betas[method_key][pname]
                avg_beta = np.mean(beta_list) if beta_list else np.nan
            else:
                total_ret = np.nan
                avg_ret = np.nan
                avg_beta = np.nan
            
            results_list.append({
                'Frequency': frequency,
                'Method': method_key,
                'Method_Name': method_name,
                'Portfolio': pname,
                'Total_Return': total_ret,
                'Avg_Period_Return': avg_ret,
                'Avg_Beta': avg_beta,
                'N_Periods': len(returns_list)
            })
    
    return pd.DataFrame(results_list), pd.DataFrame(prediction_results)


def run_full_analysis():
    """
    Run complete analysis across all frequencies.
    
    Following Hollstein et al. (2020) & Frazzini & Pedersen (2014):
    - Beta prediction accuracy (RMSE)
    - BAB portfolio returns
    
    Returns:
        Tuple of (detailed_results_df, prediction_df, summary_df)
    """
    print("\n" + "=" * 80)
    print("S&P500 STOCK LOW-BETA ANOMALY ANALYSIS")
    print("Following Hollstein et al. (2020) & Frazzini & Pedersen (2014)")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Data Source: S&P500 Constituents")
    print(f"Frequencies: {', '.join(CONFIG['frequencies'])}")
    print("Trading Hours: 9:30 AM - 4:00 PM (390 min/day)")
    print("=" * 80)
    
    # Data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Check if preprocessed data exists
    sp500_dir = data_dir / "sp500"
    if not sp500_dir.exists():
        print(f"ERROR: Preprocessed data not found: {sp500_dir}")
        print("Please run 'python preprocess_stock_data.py' first!")
        return None, None, None
    
    print(f"Using preprocessed data from: {sp500_dir}")
    
    # Collect results for all frequencies
    all_portfolio_results = []
    all_prediction_results = []
    
    for freq in CONFIG['frequencies']:
        portfolio_df, prediction_df = analyze_single_frequency(freq, data_dir)
        if not portfolio_df.empty:
            all_portfolio_results.append(portfolio_df)
        if not prediction_df.empty:
            all_prediction_results.append(prediction_df)
    
    if not all_portfolio_results:
        print("No results generated!")
        return None, None, None
    
    # Combine all results
    detailed_df = pd.concat(all_portfolio_results, ignore_index=True)
    prediction_df = pd.concat(all_prediction_results, ignore_index=True) if all_prediction_results else pd.DataFrame()
    
    # Create summary
    summary_df = create_summary(detailed_df, prediction_df)
    
    return detailed_df, prediction_df, summary_df


def create_summary(detailed_df: pd.DataFrame, prediction_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create summary statistics from detailed results."""
    
    # BAB returns pivot
    bab_df = detailed_df[detailed_df['Portfolio'] == 'BAB'].copy()
    
    summary_list = []
    
    for method in CONFIG['methods'].keys():
        method_data = bab_df[bab_df['Method'] == method]
        
        for _, row in method_data.iterrows():
            summary_list.append({
                'Method': row['Method_Name'],
                'Frequency': row['Frequency'],
                'BAB_Total_Return': row['Total_Return'],
                'BAB_Avg_Period_Return': row['Avg_Period_Return'],
                'N_Periods': row['N_Periods']
            })
    
    summary_df = pd.DataFrame(summary_list)
    
    return summary_df


def print_comprehensive_summary(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, prediction_df: pd.DataFrame = None):
    """
    Print comprehensive analysis summary.
    
    Following Hollstein et al. (2020) & Frazzini & Pedersen (2014).
    """
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("   Following Hollstein et al. (2020) & Frazzini & Pedersen (2014)")
    print("=" * 80)
    
    # 1. Parameter Settings
    print("\n" + "-" * 60)
    print("PARAMETER SETTINGS (Hollstein et al. 2020):")
    print("-" * 60)
    print(f"HF Window: {CONFIG['window_days'].get('30min', 44)} days (High-frequency)")
    print(f"Daily Window: {CONFIG['window_days'].get('1d', 252)} days")
    print(f"Beta Shrinkage: {CONFIG['beta_shrinkage']} (Vasicek)")
    print(f"Portfolios: {CONFIG['n_portfolios']} (Quintile)")
    print("Rebalancing: Monthly")
    print("BAB Formula: (1/β_L) × R_L - (1/β_H) × R_H")
    
    print("\n" + "-" * 60)
    print(f"{'Frequency':<10} {'Window (days)':<15} {'Obs/Day':<10}")
    print("-" * 60)
    
    for freq in CONFIG['frequencies']:
        window_days = CONFIG['window_days'].get(freq, 22)
        freq_minutes = STOCK_FREQUENCY_MINUTES.get(freq, 1)
        obs_per_day = STOCK_MINUTES_PER_DAY // freq_minutes
        print(f"{freq:<10} {window_days:<15} {obs_per_day:>6}")
    
    # 2. Beta Prediction Accuracy (Hollstein et al. 2020 - KEY RESULT)
    if prediction_df is not None and not prediction_df.empty:
        print("\n" + "=" * 80)
        print("BETA PREDICTION ACCURACY (Hollstein et al. 2020)")
        print("   Lower RMSE = Better prediction of future realized beta")
        print("=" * 80)
        
        rmse_pivot = prediction_df.pivot_table(
            values='RMSE',
            index='Method_Name',
            columns='Frequency',
            aggfunc='first'
        )
        
        freq_order = ['1min', '5min', '15min', '30min', '1h', '1d']
        rmse_pivot = rmse_pivot[[f for f in freq_order if f in rmse_pivot.columns]]
        
        print("\nBeta Prediction RMSE:")
        print(rmse_pivot.round(4).to_string())
        
        # Best frequency per method
        print("\n" + "-" * 60)
        print("[BEST] Frequency per Method (Lowest RMSE):")
        for method in rmse_pivot.index:
            best_freq = rmse_pivot.loc[method].idxmin()
            best_rmse = rmse_pivot.loc[method].min()
            print(f"   {method}: {best_freq} (RMSE={best_rmse:.4f})")
        
        # Compare HF vs Daily (Hollstein et al. key finding)
        if '30min' in rmse_pivot.columns and '1d' in rmse_pivot.columns:
            print("\n" + "-" * 60)
            print("HF(30min) vs Daily(1d) - Hollstein et al. Key Finding:")
            for method in rmse_pivot.index:
                hf_rmse = rmse_pivot.loc[method, '30min']
                daily_rmse = rmse_pivot.loc[method, '1d']
                improvement = (1 - hf_rmse / daily_rmse) * 100 if daily_rmse > 0 else 0
                status = "[+]" if hf_rmse < daily_rmse else "[-]"
                print(f"   {status} {method}: HF={hf_rmse:.4f}, Daily={daily_rmse:.4f} ({improvement:+.1f}%)")
    
    # 3. BAB Returns Table
    print("\n" + "=" * 80)
    print("BAB RETURNS BY METHOD & FREQUENCY")
    print("   (Leverage-adjusted: Long 1/β_L, Short 1/β_H)")
    print("=" * 80)
    
    bab_df = detailed_df[detailed_df['Portfolio'] == 'BAB'].copy()
    bab_pivot = bab_df.pivot_table(
        values='Total_Return',
        index='Method_Name',
        columns='Frequency',
        aggfunc='first'
    )
    
    # Reorder columns
    freq_order = ['1min', '5min', '15min', '30min', '1h', '1d']
    bab_pivot = bab_pivot[[f for f in freq_order if f in bab_pivot.columns]]
    
    print("\nBAB Total Returns (%):")
    print(bab_pivot.round(2).to_string())
    
    # 4. Average BAB by Method
    print("\n" + "-" * 60)
    print("Average BAB Return by Method:")
    avg_bab = bab_pivot.mean(axis=1).sort_values(ascending=False)
    for method, avg in avg_bab.items():
        status = "[+]" if avg > 0 else "[-]"
        print(f"  {status} {method}: {avg:.2f}%")
    
    # 4. Best Method per Frequency
    print("\n" + "-" * 60)
    print("[BEST] Method per Frequency:")
    for col in bab_pivot.columns:
        best_method = bab_pivot[col].idxmax()
        best_return = bab_pivot[col].max()
        status = "[+]" if best_return > 0 else "[-]"
        print(f"  {col}: {best_method} ({best_return:.2f}%) {status}")
    
    # 5. Overall Best
    print("\n" + "-" * 60)
    overall_best = avg_bab.idxmax()
    print(f"[WINNER] Overall Best Method (Avg BAB): {overall_best} ({avg_bab[overall_best]:.2f}%)")
    
    # 6. Quintile Portfolio Returns - Show ALL methods
    print("\n" + "=" * 80)
    print("QUINTILE PORTFOLIO RETURNS (Q1=Low Beta, Q5=High Beta)")
    print("=" * 80)
    
    # Show for all methods
    for method_key, method_name in CONFIG['methods'].items():
        quintile_df = detailed_df[
            (detailed_df['Method'] == method_key) & 
            (detailed_df['Portfolio'].str.startswith('Q'))
        ]
        
        if not quintile_df.empty:
            quintile_pivot = quintile_df.pivot_table(
                values='Total_Return',
                index='Portfolio',
                columns='Frequency',
                aggfunc='first'
            )
            quintile_pivot = quintile_pivot[[f for f in freq_order if f in quintile_pivot.columns]]
            print(f"\n{method_name}:")
            print(quintile_pivot.round(2).to_string())
    
    # 7. Key Findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Check for Low-Beta Anomaly
    positive_freqs = []
    for col in bab_pivot.columns:
        if bab_pivot[col].max() > 0:
            positive_freqs.append(col)
    
    if positive_freqs:
        print(f"\n[+] Low-Beta Anomaly DETECTED in: {', '.join(positive_freqs)}")
        for freq in positive_freqs:
            best = bab_pivot[freq].idxmax()
            ret = bab_pivot[freq].max()
            print(f"   → {freq}: {best} achieves +{ret:.2f}% BAB return")
    else:
        print("\n[-] No Low-Beta Anomaly detected (all BAB returns negative)")
    
    # Market characteristic notes
    print("\n" + "-" * 60)
    print("[NOTE] US Stock Market Characteristics:")
    print("   - Limited trading hours (9:30 AM - 4:00 PM)")
    print("   - Lower volatility compared to crypto")
    print("   - Established low-beta anomaly in literature")
    print("   - Weekly rebalancing may be suboptimal")
    
    # 7. Method Descriptions
    print("\n" + "=" * 80)
    print("METHOD DESCRIPTIONS")
    print("=" * 80)
    print("""
1. TARV Individual α    : Original TARV (tail-first), per-asset alpha estimation
2. TARV Rolling α       : Original TARV with rolling universal alpha across assets
3. Truncate-First Ind α : Truncation before tail estimation, per-asset alpha
4. Truncate-First Roll α: Truncation before tail estimation, rolling universal alpha
5. Simple Truncation    : Fixed 95% quantile truncation (no tail-adaptive)
6. Standard RV          : No truncation, simple realized variance/covariance

Reference: Shin, Kim & Fan (2023) - Tail-Adaptive Realized Volatility
""")


def save_results(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """Save all results to CSV files."""
    
    # Detailed results
    detailed_path = output_dir / "stock_analysis_results.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\n[SAVED] Detailed results saved to: {detailed_path}")
    
    # Summary
    summary_path = output_dir / "stock_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Summary saved to: {summary_path}")
    
    # BAB pivot table
    bab_df = detailed_df[detailed_df['Portfolio'] == 'BAB'].copy()
    bab_pivot = bab_df.pivot_table(
        values='Total_Return',
        index='Method_Name',
        columns='Frequency',
        aggfunc='first'
    )
    bab_path = output_dir / "stock_bab_returns.csv"
    bab_pivot.to_csv(bab_path)
    print(f"[SAVED] BAB returns table saved to: {bab_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    
    # Run full analysis
    detailed_df, prediction_df, summary_df = run_full_analysis()
    
    if detailed_df is None:
        print("\n[ERROR] Analysis failed!")
        return
    
    # Print comprehensive summary
    print_comprehensive_summary(detailed_df, summary_df, prediction_df)
    
    # Save results
    output_dir = Path(__file__).parent
    save_results(detailed_df, summary_df, output_dir)
    
    # Save prediction results
    if prediction_df is not None and not prediction_df.empty:
        pred_path = output_dir / "stock_beta_prediction.csv"
        prediction_df.to_csv(pred_path, index=False)
        print(f"[SAVED] Beta prediction results saved to: {pred_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
