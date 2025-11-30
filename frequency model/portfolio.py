"""
Low-Beta Anomaly Portfolio Construction

TERMINOLOGY NOTE:
================
This module uses two different concepts of "alpha":

1. TAIL ALPHA (tail_alpha): The tail index from TARV estimation
   - Measures the heaviness of return distribution tails
   - Higher values = lighter tails (closer to Gaussian)
   - Lower values = heavier tails (more extreme events)
   - Used in truncation threshold calculation
   - Can be "individual" (per-asset) or "universal" (cross-market)

2. JENSEN'S ALPHA (jensen_alpha): CAPM abnormal return  
   - α = E[R] - β × E[Rm] (excess return not explained by market)
   - Positive = outperformance vs CAPM prediction
   - The "alpha" in "Low-Beta Anomaly"
   - What we're testing: Do low-beta assets have positive jensen_alpha?

This module implements:
1. Rolling window beta estimation using TARV
2. Beta-sorted portfolio construction (deciles/quintiles)
3. Long-Short (BAB: Betting Against Beta) portfolio
4. Performance analysis and visualization
5. Individual vs Universal tail_alpha comparison

Based on: Shin, Kim, & Fan (2023)
"Measuring the Low-Beta Anomaly: High-Frequency Evidence"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

# Import TARV functions
from tarv import (
    calculate_realized_beta,
    calculate_tarv_variance,
    calculate_tarv_covariance,
    preaverage_returns,
    estimate_universal_alpha
)


@dataclass
class PortfolioWeights:
    """Portfolio weights at a given time."""
    timestamp: datetime
    weights: Dict[str, float]  # asset_id -> weight
    betas: Dict[str, float]    # asset_id -> beta
    portfolio_type: str        # 'low_beta', 'high_beta', 'long_short'


@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    avg_beta: float
    n_rebalances: int
    

class BetaEstimator:
    """
    Rolling window beta estimator using TARV.
    
    Supports universal_alpha for cross-market tail index estimation.
    """
    
    def __init__(
        self,
        window_minutes: int = 1440 * 30,  # 30 days of minute data
        min_observations: int = 1000,
        use_tarv: bool = True,
        c_omega: float = 0.333,
        verbose: bool = False,
        universal_alpha: Optional[float] = None
    ):
        """
        Initialize beta estimator.
        
        Args:
            window_minutes: Rolling window size in minutes
            min_observations: Minimum observations required
            use_tarv: If True, use TARV; else use standard RV
            c_omega: Pre-averaging parameter
            verbose: Print debug info
            universal_alpha: If provided, use this alpha for all estimations
                           (for cross-market comparison, e.g., stock + crypto)
        """
        self.window_minutes = window_minutes
        self.min_observations = min_observations
        self.use_tarv = use_tarv
        self.c_omega = c_omega
        self.verbose = verbose
        self.universal_alpha = universal_alpha
    
    def estimate_beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Estimate beta for a single asset.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
        
        Returns:
            (beta, diagnostics_dict)
        """
        n = len(asset_returns)
        
        if n < self.min_observations:
            warnings.warn(f"Only {n} observations, need {self.min_observations}")
            return np.nan, {'error': 'insufficient_data'}
        
        # Calculate K (pre-averaging window) based on c_omega
        K = max(2, int(self.c_omega * np.sqrt(n)))
        
        # Use TARV or standard
        if self.use_tarv:
            beta, info = calculate_realized_beta(
                asset_returns=asset_returns,
                market_returns=market_returns,
                K=K,
                p=2,
                use_tarv=True,
                universal_alpha=self.universal_alpha
            )
            
            diagnostics = {
                'method': 'TARV',
                'K': K,
                'cov_truncation': info.get('cov_truncation_rate', np.nan),
                'var_truncation': info.get('var_truncation_rate', np.nan),
                # TAIL ALPHA: tail index for truncation (NOT Jensen's alpha!)
                'cov_tail_alpha': info.get('cov_alpha', np.nan),
                'var_tail_alpha': info.get('var_alpha', np.nan),
                'universal_tail_alpha': self.universal_alpha,  # None if using individual
                'standard_beta': info.get('standard_cov', 0) / info.get('standard_var', 1) if info.get('standard_var', 0) != 0 else np.nan
            }
        else:
            # Standard realized beta
            cov = np.sum(asset_returns * market_returns)
            var = np.sum(market_returns ** 2)
            beta = cov / var if var != 0 else np.nan
            diagnostics = {'method': 'Standard'}
        
        return beta, diagnostics
    
    def estimate_all_betas(
        self,
        assets_data: Dict[str, np.ndarray],
        market_returns: np.ndarray
    ) -> Dict[str, Tuple[float, Dict]]:
        """
        Estimate betas for all assets.
        
        Args:
            assets_data: Dict of asset_id -> returns array
            market_returns: Market return series
        
        Returns:
            Dict of asset_id -> (beta, diagnostics)
        """
        results = {}
        
        for asset_id, asset_returns in assets_data.items():
            beta, diag = self.estimate_beta(asset_returns, market_returns)
            results[asset_id] = (beta, diag)
            
            if self.verbose:
                print(f"  {asset_id}: β = {beta:.4f}")
        
        return results


# =============================================================================
# Tail Alpha Comparison: Individual vs Universal
# =============================================================================

def compare_tail_alpha_methods(
    assets_data: Dict[str, np.ndarray],
    market_returns: np.ndarray,
    universal_tail_alpha: float,
    c_omega: float = 0.333,
    min_observations: int = 1000
) -> pd.DataFrame:
    """
    Compare beta estimates using individual vs universal tail_alpha.
    
    This helps understand the impact of using a cross-market universal
    tail index versus estimating it individually per asset.
    
    TERMINOLOGY:
    - tail_alpha: Tail index for TARV truncation (NOT Jensen's alpha!)
    - Individual: Estimate tail_alpha separately for each asset
    - Universal: Use same tail_alpha across all assets (from cross-market estimation)
    
    Args:
        assets_data: Dict of asset_id -> returns array
        market_returns: Market return series
        universal_tail_alpha: Pre-estimated universal tail index
        c_omega: Pre-averaging parameter
        min_observations: Minimum required observations
    
    Returns:
        DataFrame comparing beta estimates:
        - beta_individual: Beta using per-asset tail_alpha
        - beta_universal: Beta using universal tail_alpha
        - tail_alpha_individual: Estimated tail_alpha for this asset
        - beta_difference: Absolute difference
        - beta_pct_diff: Percentage difference
    """
    # Create estimators
    estimator_individual = BetaEstimator(
        use_tarv=True, 
        c_omega=c_omega,
        min_observations=min_observations,
        universal_alpha=None  # Individual estimation
    )
    
    estimator_universal = BetaEstimator(
        use_tarv=True,
        c_omega=c_omega, 
        min_observations=min_observations,
        universal_alpha=universal_tail_alpha
    )
    
    results = []
    
    for asset_id, asset_returns in assets_data.items():
        # Individual tail_alpha
        beta_ind, diag_ind = estimator_individual.estimate_beta(
            asset_returns, market_returns
        )
        
        # Universal tail_alpha
        beta_univ, diag_univ = estimator_universal.estimate_beta(
            asset_returns, market_returns
        )
        
        # Get individual tail_alpha (average of cov and var)
        ind_tail_alpha = np.mean([
            diag_ind.get('var_tail_alpha', np.nan),
            diag_ind.get('cov_tail_alpha', np.nan)
        ])
        
        results.append({
            'asset_id': asset_id,
            'beta_individual': beta_ind,
            'beta_universal': beta_univ,
            'tail_alpha_individual': ind_tail_alpha,
            'tail_alpha_universal': universal_tail_alpha,
            'beta_difference': abs(beta_ind - beta_univ) if not np.isnan(beta_ind) else np.nan,
            'beta_pct_diff': abs(beta_ind - beta_univ) / abs(beta_ind) * 100 if beta_ind != 0 and not np.isnan(beta_ind) else np.nan,
            'truncation_rate_ind': diag_ind.get('var_truncation', np.nan),
            'truncation_rate_univ': diag_univ.get('var_truncation', np.nan)
        })
    
    df = pd.DataFrame(results)
    
    return df


def print_tail_alpha_comparison(comparison_df: pd.DataFrame, universal_tail_alpha: float):
    """
    Print a summary of the individual vs universal tail_alpha comparison.
    
    Args:
        comparison_df: Output from compare_tail_alpha_methods()
        universal_tail_alpha: The universal tail_alpha used
    """
    print("\n" + "="*70)
    print("📊 TAIL ALPHA COMPARISON: Individual vs Universal")
    print("="*70)
    print("\n⚠️  NOTE: 'tail_alpha' here refers to the TARV tail index,")
    print("    NOT Jensen's alpha (abnormal return) from CAPM!")
    print()
    
    print(f"🌐 Universal tail_alpha used: {universal_tail_alpha:.4f}")
    print()
    
    # Summary statistics
    valid_df = comparison_df.dropna(subset=['beta_individual', 'beta_universal'])
    
    print("📈 Beta Estimation Summary:")
    print(f"   Number of assets: {len(valid_df)}")
    print(f"   Avg beta (individual): {valid_df['beta_individual'].mean():.4f}")
    print(f"   Avg beta (universal):  {valid_df['beta_universal'].mean():.4f}")
    print()
    
    print("📉 Individual tail_alpha Statistics:")
    print(f"   Mean:   {valid_df['tail_alpha_individual'].mean():.4f}")
    print(f"   Std:    {valid_df['tail_alpha_individual'].std():.4f}")
    print(f"   Min:    {valid_df['tail_alpha_individual'].min():.4f}")
    print(f"   Max:    {valid_df['tail_alpha_individual'].max():.4f}")
    print()
    
    print("🔄 Beta Difference (Individual vs Universal):")
    print(f"   Mean absolute diff:  {valid_df['beta_difference'].mean():.6f}")
    print(f"   Mean percentage diff: {valid_df['beta_pct_diff'].mean():.2f}%")
    print(f"   Max percentage diff:  {valid_df['beta_pct_diff'].max():.2f}%")
    print()
    
    # Correlation
    corr = valid_df['beta_individual'].corr(valid_df['beta_universal'])
    print(f"📊 Correlation between methods: {corr:.4f}")
    print()
    
    # Truncation comparison
    print("✂️  Truncation Rate Comparison:")
    print(f"   Individual avg: {valid_df['truncation_rate_ind'].mean():.2f}%")
    print(f"   Universal avg:  {valid_df['truncation_rate_univ'].mean():.2f}%")
    print()
    
    # Recommendation
    if valid_df['beta_pct_diff'].mean() < 5:
        print("✅ RECOMMENDATION: Methods produce similar results.")
        print("   Universal tail_alpha is suitable for cross-market comparison.")
    else:
        print("⚠️  RECOMMENDATION: Significant differences detected.")
        print("   Consider investigating which method is more appropriate.")


class PortfolioConstructor:
    """
    Construct beta-sorted portfolios.
    """
    
    def __init__(
        self,
        n_portfolios: int = 5,  # Quintiles
        weighting: str = 'equal',  # 'equal' or 'beta_weighted'
        long_short: bool = True
    ):
        """
        Initialize portfolio constructor.
        
        Args:
            n_portfolios: Number of portfolios (5=quintiles, 10=deciles)
            weighting: Weighting scheme within portfolio
            long_short: Whether to create long-short portfolio
        """
        self.n_portfolios = n_portfolios
        self.weighting = weighting
        self.long_short = long_short
    
    def construct_portfolios(
        self,
        betas: Dict[str, float],
        timestamp: datetime
    ) -> Dict[str, PortfolioWeights]:
        """
        Construct beta-sorted portfolios.
        
        Args:
            betas: Dict of asset_id -> beta
            timestamp: Current timestamp
        
        Returns:
            Dict of portfolio_name -> PortfolioWeights
        """
        # Remove NaN betas
        valid_betas = {k: v for k, v in betas.items() if not np.isnan(v)}
        n_assets = len(valid_betas)
        
        if n_assets < self.n_portfolios:
            warnings.warn(f"Only {n_assets} assets, need at least {self.n_portfolios}")
            return {}
        
        # Sort assets by beta
        sorted_assets = sorted(valid_betas.items(), key=lambda x: x[1])
        
        # Divide into portfolios
        assets_per_portfolio = n_assets // self.n_portfolios
        portfolios = {}
        
        # Portfolio naming scheme based on number of portfolios
        if self.n_portfolios == 4:
            # Quartiles: Low, Mid-Low, Mid-High, High
            portfolio_names = ["Low_Beta", "MidLow_Beta", "MidHigh_Beta", "High_Beta"]
        elif self.n_portfolios == 5:
            # Quintiles: Low, Q2, Q3, Q4, High
            portfolio_names = ["Low_Beta", "Q2", "Q3", "Q4", "High_Beta"]
        elif self.n_portfolios == 10:
            # Deciles: D1 to D10
            portfolio_names = [f"D{i+1}" for i in range(10)]
            portfolio_names[0] = "Low_Beta"
            portfolio_names[-1] = "High_Beta"
        else:
            portfolio_names = [f"P{i+1}" for i in range(self.n_portfolios)]
            portfolio_names[0] = "Low_Beta"
            portfolio_names[-1] = "High_Beta"
        
        for i in range(self.n_portfolios):
            start_idx = i * assets_per_portfolio
            if i == self.n_portfolios - 1:
                # Last portfolio gets remaining assets
                end_idx = n_assets
            else:
                end_idx = (i + 1) * assets_per_portfolio
            
            portfolio_assets = sorted_assets[start_idx:end_idx]
            
            # Calculate weights
            if self.weighting == 'equal':
                weight = 1.0 / len(portfolio_assets)
                weights = {asset: weight for asset, _ in portfolio_assets}
            elif self.weighting == 'beta_weighted':
                # Inverse beta weighting (lower beta -> higher weight)
                total_inv_beta = sum(1.0 / abs(b) for _, b in portfolio_assets if b != 0)
                weights = {
                    asset: (1.0 / abs(b)) / total_inv_beta 
                    for asset, b in portfolio_assets if b != 0
                }
            else:
                weights = {asset: 1.0 / len(portfolio_assets) for asset, _ in portfolio_assets}
            
            asset_betas = {asset: b for asset, b in portfolio_assets}
            
            portfolio_name = portfolio_names[i]
            
            portfolios[portfolio_name] = PortfolioWeights(
                timestamp=timestamp,
                weights=weights,
                betas=asset_betas,
                portfolio_type=portfolio_name
            )
        
        # Create Long-Short portfolio (BAB: Betting Against Beta)
        if self.long_short and "Low_Beta" in portfolios and "High_Beta" in portfolios:
            low_beta_weights = portfolios["Low_Beta"].weights
            high_beta_weights = portfolios["High_Beta"].weights
            
            # Long low-beta, short high-beta
            ls_weights = {}
            for asset, w in low_beta_weights.items():
                ls_weights[asset] = w  # Long
            for asset, w in high_beta_weights.items():
                ls_weights[asset] = -w  # Short
            
            all_betas = {**portfolios["Low_Beta"].betas, **portfolios["High_Beta"].betas}
            
            portfolios["BAB"] = PortfolioWeights(
                timestamp=timestamp,
                weights=ls_weights,
                betas=all_betas,
                portfolio_type="long_short"
            )
        
        return portfolios


class BacktestEngine:
    """
    Backtest beta-sorted portfolios.
    """
    
    def __init__(
        self,
        beta_estimator: BetaEstimator,
        portfolio_constructor: PortfolioConstructor,
        rebalance_frequency: str = 'monthly',  # 'daily', 'weekly', 'monthly'
        transaction_cost: float = 0.001  # 0.1%
    ):
        """
        Initialize backtest engine.
        
        Args:
            beta_estimator: BetaEstimator instance
            portfolio_constructor: PortfolioConstructor instance
            rebalance_frequency: How often to rebalance
            transaction_cost: Transaction cost per trade
        """
        self.beta_estimator = beta_estimator
        self.portfolio_constructor = portfolio_constructor
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        self.portfolio_history: List[Dict[str, PortfolioWeights]] = []
        self.returns_history: Dict[str, List[float]] = {}
        self.dates: List[datetime] = []
    
    def run_backtest(
        self,
        assets_prices: pd.DataFrame,
        market_prices: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.Series]:
        """
        Run backtest.
        
        Args:
            assets_prices: DataFrame with asset prices (columns=assets, index=datetime)
            market_prices: Series with market prices
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Dict of portfolio_name -> returns series
        """
        # Align data
        common_idx = assets_prices.index.intersection(market_prices.index)
        assets_prices = assets_prices.loc[common_idx]
        market_prices = market_prices.loc[common_idx]
        
        # Calculate returns
        assets_returns = assets_prices.pct_change().dropna()
        market_returns = market_prices.pct_change().dropna()
        
        # Align after returns calculation
        common_idx = assets_returns.index.intersection(market_returns.index)
        assets_returns = assets_returns.loc[common_idx]
        market_returns = market_returns.loc[common_idx]
        
        # Filter dates
        if start_date:
            mask = assets_returns.index >= start_date
            assets_returns = assets_returns.loc[mask]
            market_returns = market_returns.loc[mask]
        if end_date:
            mask = assets_returns.index <= end_date
            assets_returns = assets_returns.loc[mask]
            market_returns = market_returns.loc[mask]
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(assets_returns.index)
        
        print(f"📊 Backtest: {len(rebalance_dates)} rebalance periods")
        print(f"   Assets: {len(assets_returns.columns)}")
        print(f"   Date range: {assets_returns.index[0]} to {assets_returns.index[-1]}")
        
        # Initialize
        # Dynamically create portfolio return trackers based on n_portfolios
        portfolio_returns = {'Market': []}
        
        if self.portfolio_constructor.n_portfolios == 4:
            for name in ['Low_Beta', 'MidLow_Beta', 'MidHigh_Beta', 'High_Beta', 'BAB']:
                portfolio_returns[name] = []
        elif self.portfolio_constructor.n_portfolios == 5:
            for name in ['Low_Beta', 'Q2', 'Q3', 'Q4', 'High_Beta', 'BAB']:
                portfolio_returns[name] = []
        else:
            portfolio_returns['Low_Beta'] = []
            portfolio_returns['High_Beta'] = []
            portfolio_returns['BAB'] = []
        
        current_weights = None
        
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_rebal_date = rebalance_dates[i + 1]
            
            # Get estimation window data
            est_end = rebal_date
            est_start = est_end - timedelta(minutes=self.beta_estimator.window_minutes)
            
            est_mask = (assets_returns.index >= est_start) & (assets_returns.index <= est_end)
            est_assets = assets_returns.loc[est_mask]
            est_market = market_returns.loc[est_mask]
            
            if len(est_assets) < self.beta_estimator.min_observations:
                continue
            
            # Estimate betas
            assets_dict = {col: est_assets[col].values for col in est_assets.columns}
            beta_results = self.beta_estimator.estimate_all_betas(
                assets_dict, est_market.values
            )
            
            betas = {k: v[0] for k, v in beta_results.items()}
            
            # Construct portfolios
            portfolios = self.portfolio_constructor.construct_portfolios(
                betas, rebal_date
            )
            
            if not portfolios:
                continue
            
            self.portfolio_history.append(portfolios)
            
            # Calculate returns until next rebalance
            hold_mask = (assets_returns.index > rebal_date) & (assets_returns.index <= next_rebal_date)
            hold_assets = assets_returns.loc[hold_mask]
            hold_market = market_returns.loc[hold_mask]
            
            if len(hold_assets) == 0:
                continue
            
            # Portfolio returns - track ALL portfolios dynamically
            for port_name, port in portfolios.items():
                if port_name not in portfolio_returns:
                    portfolio_returns[port_name] = []
                
                port_ret = 0.0
                
                for asset, weight in port.weights.items():
                    if asset in hold_assets.columns:
                        asset_ret = (1 + hold_assets[asset]).prod() - 1
                        port_ret += weight * asset_ret
                
                portfolio_returns[port_name].append({
                    'date': next_rebal_date,
                    'return': port_ret,
                    'avg_beta': np.mean(list(port.betas.values()))
                })
            
            # Market return
            mkt_ret = (1 + hold_market).prod() - 1
            portfolio_returns['Market'].append({
                'date': next_rebal_date,
                'return': mkt_ret
            })
            
            self.dates.append(next_rebal_date)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{len(rebalance_dates)-1} periods")
        
        # Convert to DataFrames
        results = {}
        for port_name, returns_list in portfolio_returns.items():
            if returns_list:
                df = pd.DataFrame(returns_list)
                df.set_index('date', inplace=True)
                results[port_name] = df
        
        return results
    
    def _get_rebalance_dates(self, index: pd.DatetimeIndex) -> List[datetime]:
        """Get rebalance dates based on frequency."""
        if self.rebalance_frequency == 'daily':
            # Daily at market close (or end of day for crypto)
            dates = index.normalize().unique()
        elif self.rebalance_frequency == 'weekly':
            # Weekly on Monday
            dates = index[index.dayofweek == 0].normalize().unique()
        elif self.rebalance_frequency == 'monthly':
            # Monthly on first day
            dates = index[index.is_month_start].normalize().unique()
        else:
            dates = index.normalize().unique()
        
        return list(dates)
    
    def calculate_performance(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> PortfolioPerformance:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns_df: DataFrame with 'return' column
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
        
        Returns:
            PortfolioPerformance object
        """
        returns = returns_df['return'].values
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Average beta
        avg_beta = returns_df['avg_beta'].mean() if 'avg_beta' in returns_df.columns else np.nan
        
        return PortfolioPerformance(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_beta=avg_beta,
            n_rebalances=n_periods
        )


def print_performance_summary(results: Dict[str, pd.DataFrame], engine: BacktestEngine):
    """Print performance summary for all portfolios."""
    print("\n" + "=" * 70)
    print("📈 PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 70)
    
    for port_name, returns_df in results.items():
        if returns_df is None or len(returns_df) == 0:
            continue
        
        perf = engine.calculate_performance(returns_df)
        
        print(f"\n🔹 {port_name}")
        print(f"   Total Return:      {perf.total_return * 100:>8.2f}%")
        print(f"   Annualized Return: {perf.annualized_return * 100:>8.2f}%")
        print(f"   Volatility:        {perf.volatility * 100:>8.2f}%")
        print(f"   Sharpe Ratio:      {perf.sharpe_ratio:>8.2f}")
        print(f"   Max Drawdown:      {perf.max_drawdown * 100:>8.2f}%")
        if not np.isnan(perf.avg_beta):
            print(f"   Avg Beta:          {perf.avg_beta:>8.2f}")
        print(f"   # Rebalances:      {perf.n_rebalances:>8d}")
    
    # BAB Analysis (Low-Beta Anomaly)
    if 'BAB' in results and 'Market' in results:
        bab_ret = results['BAB']['return'].mean()
        mkt_ret = results['Market']['return'].mean()
        
        print("\n" + "-" * 70)
        print("🎯 LOW-BETA ANOMALY ANALYSIS")
        print("-" * 70)
        print(f"   BAB Avg Period Return:    {bab_ret * 100:>8.4f}%")
        print(f"   Market Avg Period Return: {mkt_ret * 100:>8.4f}%")
        
        if bab_ret > 0:
            print("\n   ✅ LOW-BETA ANOMALY DETECTED!")
            print("      Low-beta assets outperformed high-beta assets")
        else:
            print("\n   ❌ No low-beta anomaly in this period")
            print("      High-beta assets outperformed as CAPM predicts")


# =============================================================================
# DEMO / TEST
# =============================================================================

def run_demo():
    """
    Run a demo with simulated HIGH-FREQUENCY data (minute-level).
    
    PURPOSE: Validate that TARV methodology works correctly.
    
    Simulation setup (CAPM holds, NO anomaly):
    - α = 0 for all assets (no alpha)
    - Returns = β × market + noise
    - Expected result: BAB ≈ 0 (no anomaly)
    
    If BAB ≈ 0 in simulation → methodology is correct
    Then we can trust results from real data.
    """
    print("=" * 70)
    print("🔬 METHODOLOGY VALIDATION - TARV Beta Estimation")
    print("=" * 70)
    print("\n📌 Purpose: Verify TARV works correctly under CAPM (no anomaly)")
    print("   Simulation: α = 0 for all assets")
    print("   Expected: BAB ≈ 0 (no low-beta anomaly)")
    
    np.random.seed(42)
    
    # Simulate HIGH-FREQUENCY data (minute-level)
    # Larger sample for more reliable validation
    n_days = 250  # 2 years of trading days
    minutes_per_day = 1440
    n_minutes = n_days * minutes_per_day  # 86,400 minutes
    n_assets = 20  # 20 assets for better portfolio construction
    
    print(f"\n📊 Generating simulated data (CAPM holds)...")
    print(f"   Days: {n_days}")
    print(f"   Minutes per day: {minutes_per_day}")
    print(f"   Total observations: {n_minutes:,}")
    print(f"   Assets: {n_assets}")
    
    # Generate timestamps (minute-level)
    dates = pd.date_range(start='2024-01-01', periods=n_minutes, freq='min')
    
    # Generate market returns (minute-level)
    minute_vol = 0.015 / np.sqrt(minutes_per_day)
    market_returns_raw = np.random.normal(0.0003 / minutes_per_day, minute_vol, n_minutes)
    
    # Add some jumps to market (heavy tails)
    n_jumps = int(n_minutes * 0.001)
    jump_indices = np.random.choice(n_minutes, n_jumps, replace=False)
    jump_sizes = np.random.choice([-1, 1], n_jumps) * np.random.exponential(0.005, n_jumps)
    market_returns_raw[jump_indices] += jump_sizes
    
    market_prices = pd.Series(
        100 * np.exp(np.cumsum(market_returns_raw)),
        index=dates,
        name='Market'
    )
    
    # Generate asset returns with different betas
    # Use realistic beta distribution (normal, centered at 1.0)
    true_betas = np.random.normal(1.0, 0.35, n_assets)  # Mean=1.0, Std=0.35
    true_betas = np.clip(true_betas, 0.3, 2.0)  # Clip extreme values
    np.random.shuffle(true_betas)
    
    assets_prices = pd.DataFrame(index=dates)
    
    print("\n   Generating assets (CAPM: α = 0 for all)...")
    print(f"   Beta distribution: Normal(μ=1.0, σ=0.35), clipped to [0.3, 2.0]")
    for i, true_beta in enumerate(true_betas):
        # Asset return = β × market + noise
        # NO ALPHA (α = 0) - CAPM holds perfectly
        idio_vol = minute_vol * 1.2
        asset_returns = true_beta * market_returns_raw + np.random.normal(0, idio_vol, n_minutes)
        
        # NO alpha added - this is the key difference!
        # α = 0 means CAPM holds, no anomaly should exist
        
        # Add idiosyncratic jumps (noise, not alpha)
        n_asset_jumps = int(n_minutes * 0.0005)
        asset_jump_idx = np.random.choice(n_minutes, n_asset_jumps, replace=False)
        asset_jump_sizes = np.random.choice([-1, 1], n_asset_jumps) * np.random.exponential(0.003, n_asset_jumps)
        asset_returns[asset_jump_idx] += asset_jump_sizes
        
        asset_prices = 100 * np.exp(np.cumsum(asset_returns))
        assets_prices[f'Asset_{i+1}'] = asset_prices
        print(f"      Asset_{i+1}: β={true_beta:.2f}, α=0.0%/yr (CAPM)")
    
    print(f"\n   True betas range: [{true_betas.min():.2f}, {true_betas.max():.2f}]")
    print(f"   All alphas = 0 (CAPM assumption)")
    
    # =========================================================================
    # PART 1: Individual vs Universal Tail Alpha Comparison
    # =========================================================================
    print("\n" + "="*70)
    print("📊 PART 1: TAIL ALPHA COMPARISON (Individual vs Universal)")
    print("="*70)
    print("\n⚠️  NOTE: 'tail_alpha' = TARV tail index (NOT Jensen's alpha!)")
    
    # Prepare returns data for comparison
    asset_returns_dict = {}
    market_returns_series = np.diff(np.log(market_prices.values))
    
    for col in assets_prices.columns:
        asset_returns_dict[col] = np.diff(np.log(assets_prices[col].values))
    
    # Simulate stock + crypto for universal tail_alpha estimation
    print("\n   Simulating stock & crypto data for universal tail_alpha...")
    stock_sim = np.random.standard_t(df=5, size=len(market_returns_series)) * minute_vol
    crypto_sim = np.random.standard_t(df=3, size=len(market_returns_series)) * minute_vol * 1.5
    
    K = int(np.sqrt(len(market_returns_series)))
    universal_tail_alpha, tail_info = estimate_universal_alpha(
        [stock_sim, crypto_sim], K, method='median'
    )
    
    print(f"   Stock tail_alpha: {tail_info['individual_alphas'][0]:.4f}")
    print(f"   Crypto tail_alpha: {tail_info['individual_alphas'][1]:.4f}")
    print(f"   Universal tail_alpha (median): {universal_tail_alpha:.4f}")
    
    # Compare Individual vs Universal
    comparison_df = compare_tail_alpha_methods(
        asset_returns_dict, market_returns_series, universal_tail_alpha,
        min_observations=3000
    )
    
    # Print comparison summary
    print_tail_alpha_comparison(comparison_df, universal_tail_alpha)
    
    # Per-asset details table
    print("\n📋 Per-Asset Beta Comparison:")
    print("-" * 95)
    print(f"{'Asset':<12} {'True β':<8} {'β (Ind)':<10} {'β (Univ)':<10} {'tail_α (Ind)':<13} {'tail_α (Univ)':<13} {'Diff %':<8}")
    print("-" * 95)
    for idx, row in comparison_df.iterrows():
        asset_idx = int(row['asset_id'].split('_')[1]) - 1
        true_beta = true_betas[asset_idx] if asset_idx < len(true_betas) else np.nan
        print(f"{row['asset_id']:<12} {true_beta:<8.2f} {row['beta_individual']:<10.4f} "
              f"{row['beta_universal']:<10.4f} {row['tail_alpha_individual']:<13.4f} "
              f"{universal_tail_alpha:<13.4f} {row['beta_pct_diff']:<8.2f}")
    print("-" * 95)
    
    # =========================================================================
    # PART 2: Portfolio Backtest
    # =========================================================================
    print("\n" + "="*70)
    print("📊 PART 2: PORTFOLIO BACKTEST (Low-Beta Anomaly Test)")
    print("="*70)
    
    # Initialize components with TARV (using individual tail_alpha by default)
    beta_estimator = BetaEstimator(
        window_minutes=minutes_per_day * 14,
        min_observations=3000,
        use_tarv=True,
        c_omega=0.333,
        verbose=False
    )
    
    portfolio_constructor = PortfolioConstructor(
        n_portfolios=4,  # Quartiles (4 groups) for 20 assets → 5 assets each
        weighting='equal',
        long_short=True
    )
    
    backtest_engine = BacktestEngine(
        beta_estimator=beta_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency='monthly',
        transaction_cost=0
    )
    
    # Run backtest
    print("\n🔄 Running backtest with TARV...")
    results = backtest_engine.run_backtest(
        assets_prices=assets_prices,
        market_prices=market_prices
    )
    
    # Print results
    print_performance_summary(results, backtest_engine)
    
    # Validation check
    print("\n" + "-" * 70)
    print("🔍 METHODOLOGY VALIDATION")
    print("-" * 70)
    
    if 'BAB' in results and len(results['BAB']) > 0:
        bab_mean = results['BAB']['return'].mean() * 100
        
        print(f"   BAB Average Return: {bab_mean:+.4f}%")
        
        if abs(bab_mean) < 1.0:  # Less than 1% is close to zero
            print("\n   ✅ VALIDATION PASSED!")
            print("      BAB ≈ 0 as expected under CAPM")
            print("      → TARV methodology is working correctly")
            print("      → Ready to test on real data")
        else:
            print(f"\n   ⚠️  BAB = {bab_mean:+.2f}% (expected ≈ 0)")
            print("      This could be due to:")
            print("      - Random noise in small sample")
            print("      - Beta estimation error")
            print("      Try running with more data or different seed")
    
    print("\n" + "=" * 70)
    print("✅ Validation completed!")
    print("=" * 70)
    print("\n💡 Next step: Run on REAL data to test for actual anomaly")
    print("   python portfolio.py --real")
    
    return results


def run_real_data_analysis(
    data_dir: str = "data",
    exchange: str = "binance",
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_tarv: bool = True,
    n_portfolios: int = 5,
    rebalance_frequency: str = 'weekly',
    window_days: int = 20,
    min_observations: int = 5000
) -> Dict[str, pd.DataFrame]:
    """
    Run Low-Beta Anomaly analysis on REAL DATA.
    
    This is the main entry point for analyzing real cryptocurrency/equity data.
    
    Args:
        data_dir: Path to data directory
        exchange: 'binance', 'upbit', or 'equities'
        symbols: List of symbols (if None, use all available)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_tarv: If True, use TARV; else use standard RV
        n_portfolios: Number of beta-sorted portfolios (5=quintiles, 10=deciles)
        rebalance_frequency: 'daily', 'weekly', or 'monthly'
        window_days: Rolling window for beta estimation (days)
        min_observations: Minimum observations required per asset
    
    Returns:
        Dict of portfolio results
    
    Example:
        >>> results = run_real_data_analysis(
        ...     data_dir="data",
        ...     exchange="binance", 
        ...     symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
        ...              'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT'],
        ...     start_date="2024-01-01",
        ...     end_date="2024-06-30",
        ...     use_tarv=True,
        ...     n_portfolios=5
        ... )
    """
    from data_loader import prepare_data_for_analysis
    
    print("=" * 70)
    print("🔬 LOW-BETA ANOMALY ANALYSIS - REAL DATA")
    print("=" * 70)
    
    # Load data
    print(f"\n📥 Loading data from {exchange}...")
    try:
        assets_prices, market_prices = prepare_data_for_analysis(
            data_dir=data_dir,
            exchange=exchange,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            min_observations=min_observations
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to collect data first:")
        print("   python collector.py")
        return {}
    
    # Calculate minutes per day (crypto = 1440, equities = ~390)
    minutes_per_day = 1440 if exchange in ['binance', 'upbit'] else 390
    
    # Initialize components
    method_name = "TARV" if use_tarv else "Standard RV"
    print(f"\n🔧 Configuration:")
    print(f"   Beta estimation: {method_name}")
    print(f"   Rolling window: {window_days} days")
    print(f"   Portfolios: {n_portfolios} (quintiles)" if n_portfolios == 5 else f"   Portfolios: {n_portfolios}")
    print(f"   Rebalancing: {rebalance_frequency}")
    
    beta_estimator = BetaEstimator(
        window_minutes=minutes_per_day * window_days,
        min_observations=min_observations,
        use_tarv=use_tarv,
        c_omega=0.333,
        verbose=False
    )
    
    portfolio_constructor = PortfolioConstructor(
        n_portfolios=n_portfolios,
        weighting='equal',
        long_short=True
    )
    
    backtest_engine = BacktestEngine(
        beta_estimator=beta_estimator,
        portfolio_constructor=portfolio_constructor,
        rebalance_frequency=rebalance_frequency,
        transaction_cost=0.001
    )
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    results = backtest_engine.run_backtest(
        assets_prices=assets_prices,
        market_prices=market_prices
    )
    
    # Print results
    print_performance_summary(results, backtest_engine)
    
    # Additional analysis
    if 'BAB' in results and len(results['BAB']) > 0:
        bab_returns = results['BAB']['return']
        
        print("\n" + "-" * 70)
        print("📊 STATISTICAL SIGNIFICANCE TEST")
        print("-" * 70)
        
        # T-test: Is BAB return significantly different from 0?
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(bab_returns, 0)
        
        print(f"   BAB Mean Return: {bab_returns.mean() * 100:.4f}%")
        print(f"   BAB Std Dev:     {bab_returns.std() * 100:.4f}%")
        print(f"   T-statistic:     {t_stat:.4f}")
        print(f"   P-value:         {p_value:.4f}")
        
        if p_value < 0.05:
            if bab_returns.mean() > 0:
                print("\n   ✅ SIGNIFICANT LOW-BETA ANOMALY (p < 0.05)")
                print("      Low-beta assets significantly outperform high-beta assets!")
            else:
                print("\n   ⚠️  SIGNIFICANT but REVERSED (p < 0.05)")
                print("      High-beta assets significantly outperform low-beta assets")
        else:
            print("\n   ❌ NOT SIGNIFICANT (p >= 0.05)")
            print("      Cannot reject that BAB return = 0")
            print("      No evidence of low-beta anomaly in this sample")
    
    print("\n" + "=" * 70)
    print("✅ Analysis completed!")
    print("=" * 70)
    
    return results
