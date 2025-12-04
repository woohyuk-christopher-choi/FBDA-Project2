#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CRYPTOCURRENCY LOW-BETA ANOMALY ANALYSIS
=============================================================================

Following the methodology from:
1. Hollstein et al. (2020) - "How to estimate beta?"
   - High-frequency beta estimation improves prediction accuracy
   - Uses 6-month HF window vs 12-month daily window
   
2. Frazzini & Pedersen (2014) - "Betting Against Beta"
   - BAB = (1/beta_L) * R_L - (1/beta_H) * R_H
   - Beta shrinkage factor: 0.6

3. Shin, Kim & Fan (2023) - "Tail-Adaptive Realized Volatility"
   - TARV method for robust variance estimation

Data: Binance cryptocurrency data (15 symbols)
Frequencies: 1min, 5min, 15min, 30min, 1h, 1d
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

from data_loader import CryptoDataLoader, MarketBenchmark, DataConfig
from tarv import (
    calculate_realized_beta,
    calculate_realized_beta_truncate_first,
    calculate_realized_beta_simple_truncation,
    estimate_universal_alpha
)


# =============================================================================
# CONFIGURATION - Following Hollstein et al. (2020) & Frazzini & Pedersen (2014)
# =============================================================================

# Crypto trades 24/7: 1440 minutes per day
CRYPTO_MINUTES_PER_DAY = 1440

CRYPTO_FREQUENCY_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '1d': 1440,
}

CONFIG = {
    # Frequencies to analyze (including 1min)
    'frequencies': ['1m', '5m', '15m', '30m', '1h', '1d'],
    
    # Frequency name mapping for display
    'freq_display': {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '1d': '1d'
    },
    
    # Hollstein et al. (2020): 6 months for HF, 12 months for daily
    'window_months': {
        '1m': 6,
        '5m': 6,
        '15m': 6,
        '30m': 6,
        '1h': 6,
        '1d': 12,
    },
    
    # Window in days (crypto trades every day)
    'window_days': {
        '1m': 180,    # 6 months * 30 days
        '5m': 180,
        '15m': 180,
        '30m': 180,
        '1h': 180,
        '1d': 365,    # 12 months
    },
    
    # Minimum observations per frequency
    'min_observations': {
        '1m': 50000,
        '5m': 10000,
        '15m': 5000,
        '30m': 2000,
        '1h': 1000,
        '1d': 100,
    },
    
    # Frazzini & Pedersen (2014): Beta shrinkage toward 1
    'beta_shrinkage': 0.6,  # Vasicek shrinkage
    
    # Portfolio settings (15 cryptos = 5 quintiles x 3 each)
    'n_portfolios': 5,
    
    # Methods to compare
    'methods': {
        'TARV_Ind': 'TARV Individual alpha',
        'TARV_Roll': 'TARV Rolling alpha',
        'TruncFirst_Ind': 'Truncate-First Ind alpha',
        'TruncFirst_Roll': 'Truncate-First Roll alpha',
        'SimpleTrunc': 'Simple Truncation',
        'StandardRV': 'Standard RV',
    }
}


# =============================================================================
# DATA LOADING
# =============================================================================

def get_data_dir() -> Path:
    """Get the data directory path."""
    current = Path(__file__).parent
    data_dir = current.parent / "data"
    if not data_dir.exists():
        data_dir = current / "data"
    return data_dir


def load_crypto_data(frequency: str, exchange: str = 'binance'):
    """
    Load cryptocurrency data for analysis.
    
    Returns:
        Tuple of (assets_df, market_series)
    """
    data_dir = get_data_dir()
    
    print(f"Loading crypto data from: {data_dir}")
    
    config = DataConfig(
        data_dir=str(data_dir),
        exchange=exchange,
        frequency=frequency,
    )
    
    loader = CryptoDataLoader(config)
    available_symbols = loader.get_available_symbols()
    
    if not available_symbols:
        raise ValueError(f"No crypto data found for {exchange} at {frequency}")
    
    print(f"Available symbols: {available_symbols}")
    
    # Load all symbols
    prices_dict = {}
    for symbol in available_symbols:
        df = loader.load_symbol(symbol)
        if df is not None and 'close' in df.columns:
            prices_dict[symbol] = df['close']
    
    if not prices_dict:
        raise ValueError("No crypto price data loaded")
    
    assets_df = pd.DataFrame(prices_dict)
    assets_df = assets_df.dropna(how='all')
    
    print(f"Loaded assets: {assets_df.shape}")
    
    # Load market benchmark
    market_series = MarketBenchmark.load_benchmark_from_file(
        data_dir=str(data_dir),
        exchange=exchange,
        frequency=frequency
    )
    
    if market_series is None:
        raise ValueError(f"No benchmark data found for {exchange} at {frequency}")
    
    print(f"Loaded market: {len(market_series)} observations")
    
    # Align indices
    common_idx = assets_df.index.intersection(market_series.index)
    assets_df = assets_df.loc[common_idx]
    market_series = market_series.loc[common_idx]
    
    print(f"Date range: {assets_df.index[0]} to {assets_df.index[-1]}")
    
    return assets_df, market_series


# =============================================================================
# BETA ESTIMATION (Following Hollstein et al. 2020)
# =============================================================================

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns from prices."""
    returns = np.log(prices / prices.shift(1))
    return returns.iloc[1:]


def estimate_betas_all_methods(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    K: int,
    universal_alpha: float = None
) -> dict:
    """
    Estimate beta using all 6 methods.
    """
    betas = {}
    
    # 1. TARV with Individual alpha (universal_alpha=None -> estimate per-asset)
    try:
        beta_tarv, _ = calculate_realized_beta(
            asset_returns, market_returns,
            K=K,
            use_tarv=True,
            universal_alpha=None
        )
        betas['TARV_Ind'] = beta_tarv
    except Exception as e:
        betas['TARV_Ind'] = np.nan
    
    # 2. TARV with Rolling universal alpha
    try:
        beta_tarv_roll, _ = calculate_realized_beta(
            asset_returns, market_returns,
            K=K,
            use_tarv=True,
            universal_alpha=universal_alpha
        )
        betas['TARV_Roll'] = beta_tarv_roll
    except Exception as e:
        betas['TARV_Roll'] = np.nan
    
    # 3. Truncate-First with Individual alpha
    try:
        beta_tf_ind, _ = calculate_realized_beta_truncate_first(
            asset_returns, market_returns,
            K=K,
            universal_alpha=None
        )
        betas['TruncFirst_Ind'] = beta_tf_ind
    except Exception as e:
        betas['TruncFirst_Ind'] = np.nan
    
    # 4. Truncate-First with Rolling universal alpha
    try:
        beta_tf_roll, _ = calculate_realized_beta_truncate_first(
            asset_returns, market_returns,
            K=K,
            universal_alpha=universal_alpha
        )
        betas['TruncFirst_Roll'] = beta_tf_roll
    except Exception as e:
        betas['TruncFirst_Roll'] = np.nan
    
    # 5. Simple Truncation (fixed 95% quantile)
    try:
        beta_simple, _ = calculate_realized_beta_simple_truncation(
            asset_returns, market_returns,
            K=K,
            truncate_quantile=0.95
        )
        betas['SimpleTrunc'] = beta_simple
    except Exception as e:
        betas['SimpleTrunc'] = np.nan
    
    # 6. Standard RV (no truncation)
    try:
        cov = np.cov(asset_returns, market_returns)[0, 1]
        var_m = np.var(market_returns, ddof=1)
        betas['StandardRV'] = cov / var_m if var_m > 0 else np.nan
    except Exception as e:
        betas['StandardRV'] = np.nan
    
    return betas


def shrink_beta(beta: float, shrinkage: float = 0.6) -> float:
    """
    Apply Vasicek shrinkage (Frazzini & Pedersen 2014).
    beta_shrunk = shrinkage * beta + (1 - shrinkage) * 1.0
    """
    if np.isnan(beta):
        return np.nan
    return shrinkage * beta + (1 - shrinkage) * 1.0


# =============================================================================
# PORTFOLIO CONSTRUCTION (Following Frazzini & Pedersen 2014)
# =============================================================================

def construct_portfolios(betas: dict, n_portfolios: int = 5) -> dict:
    """
    Construct portfolios based on beta ranking.
    Q1 = Low Beta, Q5 = High Beta
    """
    valid_betas = {k: v for k, v in betas.items() if not np.isnan(v)}
    
    if len(valid_betas) < n_portfolios:
        return {}
    
    sorted_assets = sorted(valid_betas.keys(), key=lambda x: valid_betas[x])
    
    portfolios = {}
    n_assets = len(sorted_assets)
    assets_per_portfolio = n_assets // n_portfolios
    
    for i in range(n_portfolios):
        start_idx = i * assets_per_portfolio
        if i == n_portfolios - 1:
            end_idx = n_assets
        else:
            end_idx = start_idx + assets_per_portfolio
        
        if i == 0:
            label = 'Q1_Low'
        elif i == n_portfolios - 1:
            label = f'Q{n_portfolios}_High'
        else:
            label = f'Q{i+1}'
        
        portfolios[label] = sorted_assets[start_idx:end_idx]
    
    return portfolios


def calculate_portfolio_returns(
    portfolios: dict,
    asset_returns: pd.DataFrame,
    period_start,
    period_end
) -> dict:
    """Calculate equal-weighted portfolio returns for a period."""
    period_returns = asset_returns.loc[period_start:period_end]
    
    if len(period_returns) == 0:
        return {}
    
    portfolio_rets = {}
    for label, assets in portfolios.items():
        valid_assets = [a for a in assets if a in period_returns.columns]
        if valid_assets:
            port_ret = period_returns[valid_assets].sum().mean()
            portfolio_rets[label] = port_ret
    
    return portfolio_rets


def calculate_bab_return(
    portfolio_returns: dict,
    portfolio_betas: dict,
    low_label: str = 'Q1_Low',
    high_label: str = 'Q5_High'
) -> float:
    """
    Calculate BAB return (Frazzini & Pedersen 2014).
    BAB = (1/beta_L) * R_L - (1/beta_H) * R_H
    """
    if low_label not in portfolio_returns or high_label not in portfolio_returns:
        return np.nan
    if low_label not in portfolio_betas or high_label not in portfolio_betas:
        return np.nan
    
    r_low = portfolio_returns[low_label]
    r_high = portfolio_returns[high_label]
    beta_low = portfolio_betas[low_label]
    beta_high = portfolio_betas[high_label]
    
    if beta_low <= 0 or beta_high <= 0:
        return np.nan
    
    bab = (1 / beta_low) * r_low - (1 / beta_high) * r_high
    return bab


# =============================================================================
# MAIN ANALYSIS (Following Hollstein et al. 2020)
# =============================================================================

def analyze_frequency(frequency: str, exchange: str = 'binance'):
    """
    Run complete analysis for a single frequency.
    """
    freq_display = CONFIG['freq_display'].get(frequency, frequency)
    
    print(f"\n{'='*60}")
    print(f"Analyzing Frequency: {freq_display}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    try:
        assets_df, market_series = load_crypto_data(frequency, exchange)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate returns
    asset_returns = calculate_log_returns(assets_df)
    market_returns = calculate_log_returns(pd.DataFrame({'Market': market_series}))['Market']
    
    # Align
    common_idx = asset_returns.index.intersection(market_returns.index)
    asset_returns = asset_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    n_obs = len(asset_returns)
    min_obs = CONFIG['min_observations'].get(frequency, 1000)
    
    if n_obs < min_obs:
        print(f"[WARNING] Not enough observations ({n_obs} < {min_obs})")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"   Cryptos: {len(asset_returns.columns)}")
    print(f"   Observations: {n_obs:,}")
    print(f"   Date Range: {asset_returns.index[0]} to {asset_returns.index[-1]}")
    
    # Window parameters
    window_months = CONFIG['window_months'].get(frequency, 6)
    window_days = CONFIG['window_days'].get(frequency, 180)
    obs_per_day = CRYPTO_MINUTES_PER_DAY // CRYPTO_FREQUENCY_MINUTES.get(frequency, 1)
    window_obs = window_days * obs_per_day
    
    print(f"   Window: {window_months} months ({window_days} days) - Hollstein et al. (2020)")
    print(f"   Beta Shrinkage: {CONFIG['beta_shrinkage']} (Vasicek)")
    print(f"   Portfolios: {CONFIG['n_portfolios']} (Quintile)")
    print(f"   Window: {window_days} days ({window_obs:,} obs)")
    print(f"   Rebalancing: Monthly (following paper)")
    
    K = int(np.sqrt(obs_per_day))
    print(f"   K (approx): {K}")
    
    # Generate monthly rebalance dates
    dates = asset_returns.index
    rebalance_dates = []
    current_month = None
    
    for date in dates:
        month_key = (date.year, date.month)
        if month_key != current_month:
            if current_month is not None:
                rebalance_dates.append(date)
            current_month = month_key
    
    min_date = dates[0] + pd.Timedelta(days=window_days)
    rebalance_dates = [d for d in rebalance_dates if d >= min_date]
    
    if len(rebalance_dates) < 2:
        print(f"[WARNING] Not enough rebalance periods")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"\n[PROCESSING] Rebalance periods (Monthly)...")
    
    all_results = []
    prediction_results = []
    
    for i, rebal_date in enumerate(rebalance_dates[:-1]):
        est_end_idx = dates.get_loc(rebal_date)
        est_start_idx = max(0, est_end_idx - window_obs)
        
        est_returns = asset_returns.iloc[est_start_idx:est_end_idx]
        est_market = market_returns.iloc[est_start_idx:est_end_idx]
        
        if len(est_returns) < min_obs // 2:
            continue
        
        hold_start = rebal_date
        hold_end = rebalance_dates[i + 1]
        
        # Estimate universal alpha (using list format like stock analysis)
        all_ret = [est_returns[col].dropna().values for col in est_returns.columns]
        all_ret.append(est_market.values)
        K = int(np.sqrt(len(est_market)))
        try:
            universal_alpha, _ = estimate_universal_alpha(all_ret, K=K, method='median')
        except Exception:
            universal_alpha = 2.0
        
        # Estimate betas for all methods
        method_results = {method: {} for method in CONFIG['methods'].keys()}
        
        for symbol in asset_returns.columns:
            asset_ret = est_returns[symbol].dropna().values
            mkt_ret = est_market.loc[est_returns[symbol].dropna().index].values
            
            if len(asset_ret) < 100:
                continue
            
            betas = estimate_betas_all_methods(
                asset_ret, mkt_ret, K, universal_alpha
            )
            
            for method, beta in betas.items():
                if not np.isnan(beta):
                    shrunk_beta = shrink_beta(beta, CONFIG['beta_shrinkage'])
                    method_results[method][symbol] = shrunk_beta
        
        # For each method, construct portfolios and calculate returns
        for method, betas in method_results.items():
            if len(betas) < CONFIG['n_portfolios']:
                continue
            
            portfolios = construct_portfolios(betas, CONFIG['n_portfolios'])
            
            if not portfolios:
                continue
            
            portfolio_betas = {}
            for label, assets in portfolios.items():
                port_beta = np.mean([betas[a] for a in assets if a in betas])
                portfolio_betas[label] = port_beta
            
            hold_returns = asset_returns.loc[hold_start:hold_end]
            portfolio_returns = calculate_portfolio_returns(
                portfolios, hold_returns, hold_start, hold_end
            )
            
            if not portfolio_returns:
                continue
            
            n_port = CONFIG['n_portfolios']
            bab = calculate_bab_return(
                portfolio_returns, portfolio_betas,
                'Q1_Low', f'Q{n_port}_High'
            )
            
            for label, ret in portfolio_returns.items():
                all_results.append({
                    'Frequency': freq_display,
                    'Method': method,
                    'Method_Name': CONFIG['methods'][method],
                    'Period': i,
                    'Rebal_Date': rebal_date,
                    'Portfolio': label,
                    'Return': ret * 100,
                    'Beta': portfolio_betas.get(label, np.nan)
                })
            
            all_results.append({
                'Frequency': freq_display,
                'Method': method,
                'Method_Name': CONFIG['methods'][method],
                'Period': i,
                'Rebal_Date': rebal_date,
                'Portfolio': 'BAB',
                'Return': bab * 100 if not np.isnan(bab) else np.nan,
                'Beta': np.nan
            })
    
    # Beta prediction accuracy (Hollstein et al. 2020)
    print(f"   Measuring beta prediction accuracy (Hollstein et al. 2020)...")
    
    for method in CONFIG['methods'].keys():
        realized_betas = []
        predicted_betas = []
        
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            est_end_idx = dates.get_loc(rebal_date)
            est_start_idx = max(0, est_end_idx - window_obs)
            
            est_returns = asset_returns.iloc[est_start_idx:est_end_idx]
            est_market = market_returns.iloc[est_start_idx:est_end_idx]
            
            real_start = rebal_date
            real_end = rebalance_dates[i + 1]
            real_start_idx = dates.get_loc(real_start)
            real_end_idx = dates.get_loc(real_end)
            
            real_returns = asset_returns.iloc[real_start_idx:real_end_idx]
            real_market = market_returns.iloc[real_start_idx:real_end_idx]
            
            all_ret = [est_returns[col].dropna().values for col in est_returns.columns]
            all_ret.append(est_market.values)
            K = int(np.sqrt(len(est_market)))
            try:
                universal_alpha, _ = estimate_universal_alpha(all_ret, K=K, method='median')
            except Exception:
                universal_alpha = 2.0
            
            for symbol in asset_returns.columns:
                try:
                    asset_ret = est_returns[symbol].dropna().values
                    mkt_ret = est_market.loc[est_returns[symbol].dropna().index].values
                    
                    if len(asset_ret) < 100:
                        continue
                    
                    pred_betas = estimate_betas_all_methods(
                        asset_ret, mkt_ret, K, universal_alpha
                    )
                    pred_beta = pred_betas.get(method, np.nan)
                    
                    if np.isnan(pred_beta):
                        continue
                    
                    asset_ret_real = real_returns[symbol].dropna().values
                    mkt_ret_real = real_market.loc[real_returns[symbol].dropna().index].values
                    
                    if len(asset_ret_real) < 50:
                        continue
                    
                    cov = np.cov(asset_ret_real, mkt_ret_real)[0, 1]
                    var_m = np.var(mkt_ret_real, ddof=1)
                    real_beta = cov / var_m if var_m > 0 else np.nan
                    
                    if not np.isnan(real_beta):
                        predicted_betas.append(pred_beta)
                        realized_betas.append(real_beta)
                except:
                    continue
        
        if len(predicted_betas) > 10:
            rmse = np.sqrt(np.mean((np.array(predicted_betas) - np.array(realized_betas))**2))
            corr = np.corrcoef(predicted_betas, realized_betas)[0, 1]
            
            prediction_results.append({
                'Frequency': freq_display,
                'Method': method,
                'Method_Name': CONFIG['methods'][method],
                'RMSE': rmse,
                'Correlation': corr,
                'N_Predictions': len(predicted_betas)
            })
    
    n_periods = len(rebalance_dates) - 1
    print(f"   [OK] Processed {n_periods} rebalance periods (Monthly)")
    
    if prediction_results:
        print(f"\n   [RMSE] Beta Prediction Accuracy:")
        for res in prediction_results:
            print(f"      {res['Method_Name']}: RMSE={res['RMSE']:.4f}, Corr={res['Correlation']:.4f}")
    
    results_df = pd.DataFrame(all_results)
    pred_df = pd.DataFrame(prediction_results)
    
    return results_df, pred_df


def run_full_analysis(exchange: str = 'binance'):
    """Run analysis for all frequencies."""
    
    all_results = []
    all_predictions = []
    
    for freq in CONFIG['frequencies']:
        results_df, pred_df = analyze_frequency(freq, exchange)
        
        if not results_df.empty:
            all_results.append(results_df)
        if not pred_df.empty:
            all_predictions.append(pred_df)
    
    if not all_results:
        return None, None, None
    
    detailed_df = pd.concat(all_results, ignore_index=True)
    prediction_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    
    # Create summary
    summary_df = create_summary(detailed_df, prediction_df)
    
    return detailed_df, prediction_df, summary_df


def create_summary(detailed_df: pd.DataFrame, prediction_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create summary statistics."""
    
    if detailed_df.empty:
        return pd.DataFrame()
    
    # Aggregate returns by Frequency, Method, Portfolio
    summary = detailed_df.groupby(['Frequency', 'Method', 'Method_Name', 'Portfolio']).agg({
        'Return': ['mean', 'std', 'count'],
        'Beta': 'mean'
    }).reset_index()
    
    summary.columns = ['Frequency', 'Method', 'Method_Name', 'Portfolio', 
                       'Mean_Return', 'Std_Return', 'N_Periods', 'Mean_Beta']
    
    # Calculate total returns
    total_returns = detailed_df.groupby(['Frequency', 'Method', 'Method_Name', 'Portfolio'])['Return'].sum().reset_index()
    total_returns.columns = ['Frequency', 'Method', 'Method_Name', 'Portfolio', 'Total_Return']
    
    summary = summary.merge(total_returns, on=['Frequency', 'Method', 'Method_Name', 'Portfolio'])
    
    return summary


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_comprehensive_summary(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, prediction_df: pd.DataFrame = None):
    """Print comprehensive results summary."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("   Following Hollstein et al. (2020) & Frazzini & Pedersen (2014)")
    print("=" * 80)
    
    # 1. Parameter settings
    print("\n" + "-" * 60)
    print("PARAMETER SETTINGS (Hollstein et al. 2020):")
    print("-" * 60)
    print(f"HF Window: 6 months (High-frequency)")
    print(f"Daily Window: 12 months")
    print(f"Beta Shrinkage: {CONFIG['beta_shrinkage']} (Vasicek)")
    print(f"Portfolios: {CONFIG['n_portfolios']} (Quintile)")
    print(f"Rebalancing: Monthly")
    print(f"BAB Formula: (1/beta_L) x R_L - (1/beta_H) x R_H")
    print(f"Market: 24/7 trading ({CRYPTO_MINUTES_PER_DAY} min/day)")
    
    # Frequency info
    print("\n" + "-" * 60)
    print(f"{'Frequency':<10} {'Window':<15} {'Obs/Day':>10}")
    print("-" * 60)
    for freq in CONFIG['frequencies']:
        freq_display = CONFIG['freq_display'].get(freq, freq)
        window = f"{CONFIG['window_months'].get(freq, 6)} months"
        obs_per_day = CRYPTO_MINUTES_PER_DAY // CRYPTO_FREQUENCY_MINUTES.get(freq, 1)
        print(f"{freq_display:<10} {window:<15} {obs_per_day:>10}")
    
    # 2. Beta Prediction Accuracy
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
        
        # Compare HF vs Daily
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
    print("   (Leverage-adjusted: Long 1/beta_L, Short 1/beta_H)")
    print("=" * 80)
    
    bab_df = detailed_df[detailed_df['Portfolio'] == 'BAB'].copy()
    bab_pivot = bab_df.pivot_table(
        values='Return',
        index='Method_Name',
        columns='Frequency',
        aggfunc='sum'
    )
    
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
    
    # 5. Best Method per Frequency
    print("\n" + "-" * 60)
    print("[BEST] Method per Frequency:")
    for col in bab_pivot.columns:
        best_method = bab_pivot[col].idxmax()
        best_return = bab_pivot[col].max()
        status = "[+]" if best_return > 0 else "[-]"
        print(f"  {col}: {best_method} ({best_return:.2f}%) {status}")
    
    # 6. Overall Best
    print("\n" + "-" * 60)
    overall_best = avg_bab.idxmax()
    print(f"[WINNER] Overall Best Method (Avg BAB): {overall_best} ({avg_bab[overall_best]:.2f}%)")
    
    # 7. Quintile Portfolio Returns - Show ALL methods
    print("\n" + "=" * 80)
    print("QUINTILE PORTFOLIO RETURNS (Q1=Low Beta, Q5=High Beta)")
    print("=" * 80)
    
    for method_key, method_name in CONFIG['methods'].items():
        quintile_df = detailed_df[
            (detailed_df['Method'] == method_key) & 
            (detailed_df['Portfolio'].str.startswith('Q'))
        ]
        
        if not quintile_df.empty:
            quintile_pivot = quintile_df.pivot_table(
                values='Return',
                index='Portfolio',
                columns='Frequency',
                aggfunc='sum'
            )
            quintile_pivot = quintile_pivot[[f for f in freq_order if f in quintile_pivot.columns]]
            print(f"\n{method_name}:")
            print(quintile_pivot.round(2).to_string())
    
    # 8. Key Findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    positive_freqs = []
    for col in bab_pivot.columns:
        if bab_pivot[col].max() > 0:
            positive_freqs.append(col)
    
    if positive_freqs:
        print(f"\n[+] Low-Beta Anomaly DETECTED in: {', '.join(positive_freqs)}")
        for freq in positive_freqs:
            best = bab_pivot[freq].idxmax()
            ret = bab_pivot[freq].max()
            print(f"   -> {freq}: {best} achieves +{ret:.2f}% BAB return")
    else:
        print("\n[-] No Low-Beta Anomaly detected (all BAB returns negative)")
    
    print("\n" + "-" * 60)
    print("[NOTE] Cryptocurrency Market Characteristics:")
    print("   - 24/7 trading (no market close)")
    print("   - Higher volatility compared to stocks")
    print("   - Potential for higher microstructure noise")
    print("   - Different market dynamics from traditional assets")
    
    # 9. Method Descriptions
    print("\n" + "=" * 80)
    print("METHOD DESCRIPTIONS")
    print("=" * 80)
    print("""
1. TARV Individual alpha    : Original TARV (tail-first), per-asset alpha estimation
2. TARV Rolling alpha       : Original TARV with rolling universal alpha across assets
3. Truncate-First Ind alpha : Truncation before tail estimation, per-asset alpha
4. Truncate-First Roll alpha: Truncation before tail estimation, rolling universal alpha
5. Simple Truncation        : Fixed 95% quantile truncation (no tail-adaptive)
6. Standard RV              : No truncation, simple realized variance/covariance

Reference: Shin, Kim & Fan (2023) - Tail-Adaptive Realized Volatility
""")


def save_results(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """Save all results to CSV files."""
    
    detailed_path = output_dir / "crypto_analysis_results.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\n[SAVED] Detailed results saved to: {detailed_path}")
    
    summary_path = output_dir / "crypto_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Summary saved to: {summary_path}")
    
    bab_df = detailed_df[detailed_df['Portfolio'] == 'BAB'].copy()
    bab_pivot = bab_df.pivot_table(
        values='Return',
        index='Method_Name',
        columns='Frequency',
        aggfunc='sum'
    )
    bab_path = output_dir / "crypto_bab_returns.csv"
    bab_pivot.to_csv(bab_path)
    print(f"[SAVED] BAB returns table saved to: {bab_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import time
    start_time = time.time()
    
    print("=" * 80)
    print("CRYPTOCURRENCY LOW-BETA ANOMALY ANALYSIS")
    print("Following Hollstein et al. (2020) & Frazzini & Pedersen (2014)")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Source: Binance Cryptocurrency")
    print(f"Frequencies: {', '.join(CONFIG['freq_display'].values())}")
    print(f"Trading Hours: 24/7 ({CRYPTO_MINUTES_PER_DAY} min/day)")
    print("=" * 80)
    
    # Run full analysis
    detailed_df, prediction_df, summary_df = run_full_analysis(exchange='binance')
    
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
        pred_path = output_dir / "crypto_beta_prediction.csv"
        prediction_df.to_csv(pred_path, index=False)
        print(f"[SAVED] Beta prediction results saved to: {pred_path}")
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! (Elapsed Time: {elapsed_time:.2f} seconds)")
    print("=" * 80)


if __name__ == "__main__":
    main()
