#!/usr/bin/env python3
"""
Visualization Module for Time Series Anomaly Analysis

Generates comprehensive visualizations to interpret:
1. How anomaly strength varies across frequencies
2. TARV vs Standard RV comparison
3. Market efficiency across time horizons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import pickle


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(filepath: Path) -> List[Dict]:
    """Load saved results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def extract_comparison_data(all_results: List[Dict]) -> pd.DataFrame:
    """Extract data for comparison across frequencies."""
    data = []

    for result in all_results:
        freq = result['frequency']
        summary = result['summary']

        # TARV metrics
        if 'bab_mean' in summary['tarv']:
            data.append({
                'frequency': freq,
                'method': 'TARV',
                'bab_return': summary['tarv']['bab_mean'] * 100,
                'bab_std': summary['tarv']['bab_std'] * 100,
                'bab_sharpe': summary['tarv']['bab_sharpe'],
                't_stat': summary['tarv']['bab_t_stat'],
                'p_value': summary['tarv']['bab_p_value'],
                'spread_mean': summary['tarv'].get('spread_mean', np.nan) * 100,
                'spread_std': summary['tarv'].get('spread_std', np.nan) * 100
            })

        # Standard RV metrics
        if 'bab_mean' in summary['standard']:
            data.append({
                'frequency': freq,
                'method': 'Standard RV',
                'bab_return': summary['standard']['bab_mean'] * 100,
                'bab_std': summary['standard']['bab_std'] * 100,
                'bab_sharpe': summary['standard']['bab_sharpe'],
                't_stat': summary['standard']['bab_t_stat'],
                'p_value': summary['standard']['bab_p_value'],
                'spread_mean': summary['standard'].get('spread_mean', np.nan) * 100,
                'spread_std': summary['standard'].get('spread_std', np.nan) * 100
            })

    return pd.DataFrame(data)


def plot_comprehensive_comparison(df: pd.DataFrame, save_path: Path = None):
    """Create comprehensive comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Low-Volatility Anomaly Across Time Series: Comprehensive Analysis',
                 fontsize=16, fontweight='bold')

    frequencies = df['frequency'].unique()
    freq_order = ['1m', '5m', '15m', '30m', '1h', '1d']
    freq_order = [f for f in freq_order if f in frequencies]

    # Plot 1: BAB Returns across frequencies
    ax1 = axes[0, 0]
    tarv_data = df[df['method'] == 'TARV'].set_index('frequency').loc[freq_order]
    std_data = df[df['method'] == 'Standard RV'].set_index('frequency').loc[freq_order]

    x = np.arange(len(freq_order))
    width = 0.35

    bars1 = ax1.bar(x - width/2, tarv_data['bab_return'], width, label='TARV',
                    color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, std_data['bab_return'], width, label='Standard RV',
                    color='coral', alpha=0.8)

    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('BAB Mean Return (%)')
    ax1.set_title('BAB Portfolio Returns Across Frequencies')
    ax1.set_xticks(x)
    ax1.set_xticklabels(freq_order)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # Add significance stars
    for i, (idx, row) in enumerate(tarv_data.iterrows()):
        if row['p_value'] < 0.01:
            ax1.text(i - width/2, row['bab_return'], '***', ha='center', va='bottom')
        elif row['p_value'] < 0.05:
            ax1.text(i - width/2, row['bab_return'], '**', ha='center', va='bottom')

    for i, (idx, row) in enumerate(std_data.iterrows()):
        if row['p_value'] < 0.01:
            ax1.text(i + width/2, row['bab_return'], '***', ha='center', va='bottom')
        elif row['p_value'] < 0.05:
            ax1.text(i + width/2, row['bab_return'], '**', ha='center', va='bottom')

    # Plot 2: T-statistics
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, tarv_data['t_stat'], width, label='TARV',
                    color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, std_data['t_stat'], width, label='Standard RV',
                    color='coral', alpha=0.8)

    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('t-statistic')
    ax2.set_title('Statistical Significance Across Frequencies')
    ax2.set_xticks(x)
    ax2.set_xticklabels(freq_order)
    ax2.legend()
    ax2.axhline(y=1.96, color='red', linestyle='--', linewidth=1, label='95% threshold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sharpe Ratios
    ax3 = axes[0, 2]
    bars1 = ax3.bar(x - width/2, tarv_data['bab_sharpe'], width, label='TARV',
                    color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, std_data['bab_sharpe'], width, label='Standard RV',
                    color='coral', alpha=0.8)

    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Performance Across Frequencies')
    ax3.set_xticks(x)
    ax3.set_xticklabels(freq_order)
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Low-High Spread
    ax4 = axes[1, 0]
    bars1 = ax4.bar(x - width/2, tarv_data['spread_mean'], width, label='TARV',
                    color='steelblue', alpha=0.8,
                    yerr=tarv_data['spread_std'], capsize=5)
    bars2 = ax4.bar(x + width/2, std_data['spread_mean'], width, label='Standard RV',
                    color='coral', alpha=0.8,
                    yerr=std_data['spread_std'], capsize=5)

    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Low-High Beta Spread (%)')
    ax4.set_title('Low-High Beta Portfolio Spread')
    ax4.set_xticks(x)
    ax4.set_xticklabels(freq_order)
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)

    # Plot 5: TARV vs Standard RV Direct Comparison
    ax5 = axes[1, 1]
    tarv_returns = tarv_data['bab_return'].values
    std_returns = std_data['bab_return'].values

    ax5.plot(freq_order, tarv_returns, marker='o', linewidth=2, markersize=8,
             label='TARV', color='steelblue')
    ax5.plot(freq_order, std_returns, marker='s', linewidth=2, markersize=8,
             label='Standard RV', color='coral')

    ax5.set_xlabel('Frequency')
    ax5.set_ylabel('BAB Mean Return (%)')
    ax5.set_title('TARV vs Standard RV: Direct Comparison')
    ax5.legend()
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Noise Reduction Effect
    ax6 = axes[1, 2]
    diff = std_returns - tarv_returns
    pct_change = (diff / np.abs(std_returns) * 100)
    colors = ['green' if d > 0 else 'red' for d in diff]

    bars = ax6.bar(freq_order, diff, color=colors, alpha=0.6)
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Standard RV - TARV (%)')
    ax6.set_title('Noise Reduction Effect\n(Positive = TARV Reduces Anomaly)')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.grid(True, alpha=0.3)

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, pct_change)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive plot saved to: {save_path}")

    plt.show()


def plot_heatmap_analysis(df: pd.DataFrame, save_path: Path = None):
    """Create heatmap showing anomaly strength across frequencies and methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Heatmap Analysis: Anomaly Characteristics Across Frequencies',
                 fontsize=14, fontweight='bold')

    frequencies = ['1m', '5m', '15m', '30m', '1h', '1d']
    frequencies = [f for f in frequencies if f in df['frequency'].values]
    methods = ['TARV', 'Standard RV']

    # Metric 1: BAB Returns
    returns_matrix = np.zeros((len(methods), len(frequencies)))
    for i, method in enumerate(methods):
        for j, freq in enumerate(frequencies):
            val = df[(df['method'] == method) & (df['frequency'] == freq)]['bab_return'].values
            returns_matrix[i, j] = val[0] if len(val) > 0 else np.nan

    ax1 = axes[0]
    sns.heatmap(returns_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=frequencies, yticklabels=methods, ax=ax1,
                cbar_kws={'label': 'Return (%)'})
    ax1.set_title('BAB Mean Returns (%)')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Method')

    # Metric 2: T-statistics
    tstat_matrix = np.zeros((len(methods), len(frequencies)))
    for i, method in enumerate(methods):
        for j, freq in enumerate(frequencies):
            val = df[(df['method'] == method) & (df['frequency'] == freq)]['t_stat'].values
            tstat_matrix[i, j] = val[0] if len(val) > 0 else np.nan

    ax2 = axes[1]
    sns.heatmap(tstat_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=frequencies, yticklabels=methods, ax=ax2,
                cbar_kws={'label': 't-statistic'})
    ax2.set_title('T-Statistics')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Method')

    # Metric 3: Sharpe Ratios
    sharpe_matrix = np.zeros((len(methods), len(frequencies)))
    for i, method in enumerate(methods):
        for j, freq in enumerate(frequencies):
            val = df[(df['method'] == method) & (df['frequency'] == freq)]['bab_sharpe'].values
            sharpe_matrix[i, j] = val[0] if len(val) > 0 else np.nan

    ax3 = axes[2]
    sns.heatmap(sharpe_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=frequencies, yticklabels=methods, ax=ax3,
                cbar_kws={'label': 'Sharpe Ratio'})
    ax3.set_title('Sharpe Ratios')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Method')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Heatmap saved to: {save_path}")

    plt.show()


def create_summary_report(df: pd.DataFrame, save_path: Path = None):
    """Generate a text summary report."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("LOW-VOLATILITY ANOMALY ACROSS TIME SERIES: SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)

    tarv_df = df[df['method'] == 'TARV']
    std_df = df[df['method'] == 'Standard RV']

    report_lines.append(f"\nTARV Method:")
    report_lines.append(f"  Average BAB Return:  {tarv_df['bab_return'].mean():.3f}%")
    report_lines.append(f"  Average Sharpe:      {tarv_df['bab_sharpe'].mean():.3f}")
    report_lines.append(f"  Significant (p<0.05): {(tarv_df['p_value'] < 0.05).sum()}/{len(tarv_df)}")

    report_lines.append(f"\nStandard RV Method:")
    report_lines.append(f"  Average BAB Return:  {std_df['bab_return'].mean():.3f}%")
    report_lines.append(f"  Average Sharpe:      {std_df['bab_sharpe'].mean():.3f}")
    report_lines.append(f"  Significant (p<0.05): {(std_df['p_value'] < 0.05).sum()}/{len(std_df)}")

    # Frequency-specific analysis
    report_lines.append("\n")
    report_lines.append("FREQUENCY-SPECIFIC ANALYSIS")
    report_lines.append("-" * 80)

    for freq in df['frequency'].unique():
        freq_data = df[df['frequency'] == freq]
        tarv_data = freq_data[freq_data['method'] == 'TARV'].iloc[0]
        std_data = freq_data[freq_data['method'] == 'Standard RV'].iloc[0]

        report_lines.append(f"\n{freq}:")
        report_lines.append(f"  TARV:        Return={tarv_data['bab_return']:.3f}%, "
                          f"t={tarv_data['t_stat']:.2f}, p={tarv_data['p_value']:.4f}")
        report_lines.append(f"  Standard RV: Return={std_data['bab_return']:.3f}%, "
                          f"t={std_data['t_stat']:.2f}, p={std_data['p_value']:.4f}")

        diff = std_data['bab_return'] - tarv_data['bab_return']
        report_lines.append(f"  Difference:  {diff:.3f}% (Standard RV - TARV)")

    # Key findings
    report_lines.append("\n")
    report_lines.append("KEY FINDINGS")
    report_lines.append("-" * 80)

    # Finding 1: Noise reduction
    avg_diff = (std_df['bab_return'] - tarv_df['bab_return']).mean()
    if avg_diff > 0:
        report_lines.append(f"\n1. NOISE CONTAMINATION: Standard RV shows {avg_diff:.3f}% higher returns on average")
        report_lines.append("   → Standard RV may be contaminated by noise/jumps")
        report_lines.append("   → TARV provides better noise filtering")
    else:
        report_lines.append(f"\n1. NOISE EFFECT: TARV shows {-avg_diff:.3f}% higher returns on average")
        report_lines.append("   → TARV may be over-filtering or removing true signal")

    # Finding 2: Persistence
    tarv_sig = (tarv_df['p_value'] < 0.05).sum()
    total = len(tarv_df)
    if tarv_sig >= total * 0.5:
        report_lines.append(f"\n2. PERSISTENCE: Anomaly significant in {tarv_sig}/{total} frequencies")
        report_lines.append("   → Suggests structural mispricing or risk factor")
    else:
        report_lines.append(f"\n2. PERSISTENCE: Anomaly significant in only {tarv_sig}/{total} frequencies")
        report_lines.append("   → Time-scale dependent or sample-specific phenomenon")

    # Finding 3: Best frequency
    best_freq = tarv_df.loc[tarv_df['t_stat'].idxmax(), 'frequency']
    best_t = tarv_df['t_stat'].max()
    report_lines.append(f"\n3. OPTIMAL FREQUENCY: Strongest anomaly at {best_freq} (t={best_t:.2f})")

    report_lines.append("\n" + "=" * 80)

    report_text = "\n".join(report_lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Summary report saved to: {save_path}")

    print(report_text)

    return report_text


def main():
    """Generate all visualizations from saved results."""
    script_dir = Path(__file__).parent
    results_file = script_dir / "time_series_anomaly_results.pkl"

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Please run time_series_anomaly.py first to generate results.")
        return

    print("📊 Loading results...")
    all_results = load_results(results_file)

    print(f"✅ Loaded results for {len(all_results)} frequencies")

    # Extract comparison data
    df = extract_comparison_data(all_results)

    # Generate visualizations
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    print("\n📈 Generating visualizations...")

    # 1. Comprehensive comparison
    plot_comprehensive_comparison(
        df,
        save_path=output_dir / "comprehensive_comparison.png"
    )

    # 2. Heatmap analysis
    plot_heatmap_analysis(
        df,
        save_path=output_dir / "heatmap_analysis.png"
    )

    # 3. Summary report
    create_summary_report(
        df,
        save_path=output_dir / "summary_report.txt"
    )

    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
