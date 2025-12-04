#!/usr/bin/env python3
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

from stock_data_loader import load_sp500_data, resample_to_frequency
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
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1h': 60,
    '1d': 390
}

CONFIG = {
    'frequencies': ['5min', '15min', '30min', '1h', '1d'],
    'window_days': {
        '5min': 30,
        '15min': 30,
        '30min': 30,
        '1h': 30,
        '1d': 120
    },
    'holding_period_days': 7,
    'min_observations': {
        '5min': 5000,
        '15min': 2000,
        '30min': 1000,
        '1h': 500,
        '1d': 50
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


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stock_data_for_frequency(
    data_file: Path,
    frequency: str
) -> tuple:
    """
    Load and resample S&P500 data for a specific frequency.
    
    Args:
        data_file: Path to Excel data file
        frequency: Target frequency ('5min', '15min', '30min', '1h', '1d')
    
    Returns:
        Tuple of (stock_prices, market_prices)
    """
    # Load raw data
    stock_prices, market_prices = load_sp500_data(str(data_file))
    
    # Resample to target frequency
    stock_resampled = resample_to_frequency(stock_prices, frequency)
    market_resampled = resample_to_frequency(
        market_prices.to_frame(), 
        frequency
    ).iloc[:, 0]
    
    # Align indices
    common_idx = stock_resampled.index.intersection(market_resampled.index)
    stock_resampled = stock_resampled.loc[common_idx]
    market_resampled = market_resampled.loc[common_idx]
    
    return stock_resampled, market_resampled


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_frequency(
    frequency: str,
    data_file: Path
) -> pd.DataFrame:
    """
    Analyze all 6 methods for a single frequency.
    
    Returns DataFrame with columns:
        Frequency, Method, Portfolio, Total_Return, Avg_Period_Return, N_Periods
    """
    print(f"\n{'='*60}")
    print(f"📊 Analyzing Frequency: {frequency}")
    print('='*60)
    
    # Get config
    window_days = CONFIG['window_days'].get(frequency, 30)
    holding_days = CONFIG['holding_period_days']
    
    # Load data
    print("📥 Loading data...")
    try:
        stock_prices, market_prices = load_stock_data_for_frequency(data_file, frequency)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # Calculate returns
    stock_returns = stock_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    
    # Align
    common_idx = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    n_stocks = len(stock_returns.columns)
    n_obs = len(stock_returns)
    
    print(f"   Stocks: {n_stocks}")
    print(f"   Observations: {n_obs:,}")
    print(f"   Date Range: {stock_returns.index[0]} to {stock_returns.index[-1]}")
    
    # Check minimum observations
    min_obs = CONFIG['min_observations'].get(frequency, 1000)
    if n_obs < min_obs:
        print(f"⚠️ Not enough observations ({n_obs} < {min_obs})")
    
    # Calculate window sizes (US market: 390 min/day)
    freq_minutes = STOCK_FREQUENCY_MINUTES.get(frequency, 1)
    obs_per_day = STOCK_MINUTES_PER_DAY // freq_minutes
    window_obs = obs_per_day * window_days
    hold_obs = obs_per_day * holding_days
    
    # For daily data, ensure K >= 3
    if frequency == '1d':
        K_base = max(3, int(0.333 * np.sqrt(window_obs)))
        min_asset_obs = 20
    else:
        K_base = max(2, int(0.333 * np.sqrt(window_obs)))
        min_asset_obs = 50
    
    print(f"   Window: {window_days} days ({window_obs:,} obs)")
    print(f"   Hold: {holding_days} days ({hold_obs:,} obs)")
    print(f"   K (approx): {K_base}")
    
    # Get rebalance points (weekly on Monday)
    rebalance_dates = pd.date_range(
        start=stock_returns.index[0],
        end=stock_returns.index[-1],
        freq='W-MON'
    )
    
    # Results storage
    methods = CONFIG['methods']
    all_results = {m: {p: [] for p in ['Low_Beta', 'Q2', 'Q3', 'Q4', 'High_Beta', 'BAB', 'Market']}
                   for m in methods.keys()}
    
    # Store all betas for later analysis
    all_betas = {m: [] for m in methods.keys()}
    
    # Process each rebalance period
    n_periods = 0
    min_obs_required = max(20, int(window_obs * 0.3))
    
    print(f"\n🔄 Processing rebalance periods...")
    
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
            if betas:
                all_betas[method_key].append(betas)
        
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
            if len(betas) < 5:
                continue
            
            # Sort by beta
            sorted_assets = sorted(betas.items(), key=lambda x: x[1])
            n_assets_port = len(sorted_assets)
            per_port = n_assets_port // 5
            
            port_names = ['Low_Beta', 'Q2', 'Q3', 'Q4', 'High_Beta']
            port_returns = {}
            
            for j, port_name in enumerate(port_names):
                start = j * per_port
                end = n_assets_port if j == 4 else (j + 1) * per_port
                port_assets = [a for a, _ in sorted_assets[start:end]]
                
                weight = 1.0 / len(port_assets)
                port_ret = 0
                valid_count = 0
                for asset in port_assets:
                    if asset in period_returns.index:
                        r = period_returns[asset]
                        if not np.isnan(r):
                            port_ret += weight * r
                            valid_count += 1
                
                if valid_count > 0:
                    port_ret = port_ret * len(port_assets) / valid_count
                
                port_returns[port_name] = port_ret
                all_results[method_key][port_name].append(port_ret)
            
            # BAB = Low_Beta - High_Beta
            bab_ret = port_returns['Low_Beta'] - port_returns['High_Beta']
            all_results[method_key]['BAB'].append(bab_ret)
            all_results[method_key]['Market'].append(market_ret)
    
    print(f"   ✅ Processed {n_periods} rebalance periods")
    
    # Aggregate results into DataFrame
    results_list = []
    
    for method_key, method_name in methods.items():
        for port_name in ['Low_Beta', 'Q2', 'Q3', 'Q4', 'High_Beta', 'BAB', 'Market']:
            returns_list = all_results[method_key][port_name]
            if returns_list:
                total_ret = float((1 + pd.Series(returns_list)).prod() - 1) * 100
                avg_ret = np.mean(returns_list) * 100
            else:
                total_ret = np.nan
                avg_ret = np.nan
            
            results_list.append({
                'Frequency': frequency,
                'Method': method_key,
                'Method_Name': method_name,
                'Portfolio': port_name,
                'Total_Return': total_ret,
                'Avg_Period_Return': avg_ret,
                'N_Periods': len(returns_list)
            })
    
    return pd.DataFrame(results_list)


def run_full_analysis():
    """
    Run complete analysis across all frequencies.
    
    Returns:
        Tuple of (detailed_results_df, summary_df)
    """
    print("\n" + "=" * 80)
    print("🚀 S&P500 STOCK LOW-BETA ANOMALY ANALYSIS")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Data Source: S&P500 Constituents")
    print(f"📈 Frequencies: {', '.join(CONFIG['frequencies'])}")
    print(f"🕐 Trading Hours: 9:30 AM - 4:00 PM (390 min/day)")
    print("=" * 80)
    
    # Find data file
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Look for S&P500 data file
    data_files = list(data_dir.glob("S&P500*.xlsx"))
    if not data_files:
        data_files = list(data_dir.glob("*S&P500*.xlsx"))
    
    if not data_files:
        print(f"❌ No S&P500 data file found in: {data_dir}")
        print("   Expected file pattern: S&P500*.xlsx")
        return None, None
    
    data_file = data_files[0]
    print(f"📂 Using data file: {data_file.name}")
    
    # Collect results for all frequencies
    all_results = []
    
    for freq in CONFIG['frequencies']:
        result_df = analyze_single_frequency(freq, data_file)
        if not result_df.empty:
            all_results.append(result_df)
    
    if not all_results:
        print("❌ No results generated!")
        return None, None
    
    # Combine all results
    detailed_df = pd.concat(all_results, ignore_index=True)
    
    # Create summary
    summary_df = create_summary(detailed_df)
    
    return detailed_df, summary_df


def create_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
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


def print_comprehensive_summary(detailed_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print comprehensive analysis summary."""
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    
    # 1. Parameter Settings
    print("\n" + "-" * 60)
    print("⚙️  PARAMETER SETTINGS BY FREQUENCY:")
    print("-" * 60)
    print(f"{'Frequency':<10} {'Window':<12} {'Window (obs)':<15} {'Hold':<10} {'K':<8}")
    print("-" * 60)
    
    for freq in CONFIG['frequencies']:
        window_days = CONFIG['window_days'].get(freq, 30)
        freq_minutes = STOCK_FREQUENCY_MINUTES.get(freq, 1)
        obs_per_day = STOCK_MINUTES_PER_DAY // freq_minutes
        window_obs = obs_per_day * window_days
        K_approx = max(3 if freq == '1d' else 2, int(0.333 * np.sqrt(window_obs)))
        print(f"{freq:<10} {window_days} days{'':<5} {window_obs:>12,} {CONFIG['holding_period_days']} days{'':<3} {K_approx:<8}")
    
    print("-" * 60)
    print("Rebalance: Weekly (every Monday)")
    print("Portfolio: 5 quintiles (Low_Beta, Q2, Q3, Q4, High_Beta)")
    print("BAB Strategy: Long Low_Beta, Short High_Beta")
    print("Trading Hours: 9:30 AM - 4:00 PM (390 min/day)")
    
    # 2. BAB Returns Table
    print("\n" + "=" * 80)
    print("📈 BAB RETURNS BY METHOD & FREQUENCY")
    print("=" * 80)
    
    bab_df = detailed_df[detailed_df['Portfolio'] == 'BAB'].copy()
    bab_pivot = bab_df.pivot_table(
        values='Total_Return',
        index='Method_Name',
        columns='Frequency',
        aggfunc='first'
    )
    
    # Reorder columns
    freq_order = ['5min', '15min', '30min', '1h', '1d']
    bab_pivot = bab_pivot[[f for f in freq_order if f in bab_pivot.columns]]
    
    print("\nBAB (Low_Beta - High_Beta) Total Returns (%):")
    print(bab_pivot.round(2).to_string())
    
    # 3. Average BAB by Method
    print("\n" + "-" * 60)
    print("📊 Average BAB Return by Method:")
    avg_bab = bab_pivot.mean(axis=1).sort_values(ascending=False)
    for method, avg in avg_bab.items():
        print(f"  {method}: {avg:.2f}%")
    
    # 4. Best Method per Frequency
    print("\n" + "-" * 60)
    print("🏆 Best Method per Frequency:")
    for col in bab_pivot.columns:
        best_method = bab_pivot[col].idxmax()
        best_return = bab_pivot[col].max()
        status = "✅ Positive" if best_return > 0 else "❌ Negative"
        print(f"  {col}: {best_method} ({best_return:.2f}%) {status}")
    
    # 5. Overall Best
    print("\n" + "-" * 60)
    overall_best = avg_bab.idxmax()
    print(f"🥇 Overall Best Method (Avg BAB): {overall_best} ({avg_bab[overall_best]:.2f}%)")
    
    # 6. Comparison: Crypto vs Stock Expected
    print("\n" + "=" * 80)
    print("📝 KEY FINDINGS")
    print("=" * 80)
    
    # Check for Low-Beta Anomaly
    positive_freqs = []
    for col in bab_pivot.columns:
        if bab_pivot[col].max() > 0:
            positive_freqs.append(col)
    
    if positive_freqs:
        print(f"\n✅ Low-Beta Anomaly DETECTED in: {', '.join(positive_freqs)}")
        for freq in positive_freqs:
            best = bab_pivot[freq].idxmax()
            ret = bab_pivot[freq].max()
            print(f"   → {freq}: {best} achieves +{ret:.2f}% BAB return")
    else:
        print("\n❌ No Low-Beta Anomaly detected (all BAB returns negative)")
    
    # Market characteristic notes
    print("\n" + "-" * 60)
    print("📌 US Stock Market Characteristics:")
    print("   - Limited trading hours (9:30 AM - 4:00 PM)")
    print("   - Lower volatility compared to crypto")
    print("   - Established low-beta anomaly in literature")
    print("   - Weekly rebalancing may be suboptimal")
    
    # 7. Method Descriptions
    print("\n" + "=" * 80)
    print("📖 METHOD DESCRIPTIONS")
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
    print(f"\n✅ Detailed results saved to: {detailed_path}")
    
    # Summary
    summary_path = output_dir / "stock_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")
    
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
    print(f"✅ BAB returns table saved to: {bab_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    
    # Run full analysis
    detailed_df, summary_df = run_full_analysis()
    
    if detailed_df is None:
        print("\n❌ Analysis failed!")
        return
    
    # Print comprehensive summary
    print_comprehensive_summary(detailed_df, summary_df)
    
    # Save results
    output_dir = Path(__file__).parent
    save_results(detailed_df, summary_df, output_dir)
    
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
