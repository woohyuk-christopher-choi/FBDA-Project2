"""
Low-Beta Anomaly Portfolio Construction

This module implements:
1. Rolling window beta estimation using TARV
2. Beta-sorted portfolio construction (quintiles/quartiles)
3. Long-Short (BAB: Betting Against Beta) portfolio
4. Backtesting and performance analysis

Based on: Shin, Kim, & Fan (2023)
"Adaptive robust large volatility matrix estimation"
Journal of Econometrics, 237(1), 105514
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from tarv import calculate_realized_beta, estimate_universal_alpha


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PortfolioWeights:
    """Portfolio weights at a given time."""
    timestamp: datetime
    weights: Dict[str, float]
    betas: Dict[str, float]
    portfolio_type: str


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


# =============================================================================
# Beta Estimator
# =============================================================================

class BetaEstimator:
    """Rolling window beta estimator using TARV."""
    
    def __init__(
        self,
        window_observations: int = 1440 * 30,  # Number of observations (not minutes)
        min_observations: int = 1000,
        use_tarv: bool = True,
        c_omega: float = 0.333,
        verbose: bool = False,
        use_rolling_alpha: bool = False  # If True, estimate alpha from window data
    ):
        self.window_observations = window_observations  # Renamed from window_minutes
        self.min_observations = min_observations
        self.use_tarv = use_tarv
        self.c_omega = c_omega
        self.verbose = verbose
        self.use_rolling_alpha = use_rolling_alpha
        self._current_alpha = None  # Store current window's alpha
    
    def estimate_rolling_alpha(
        self,
        assets_data: Dict[str, np.ndarray],
        market_returns: np.ndarray
    ) -> float:
        """Estimate universal alpha from current window data (no look-ahead)."""
        all_returns = list(assets_data.values()) + [market_returns]
        K = max(2, int(self.c_omega * np.sqrt(len(market_returns))))
        
        universal_alpha, info = estimate_universal_alpha(all_returns, K=K, method='median')
        self._current_alpha = universal_alpha
        
        if self.verbose:
            print(f"   Rolling α = {universal_alpha:.4f}")
        
        return universal_alpha
    
    def estimate_beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        universal_alpha: float = None
    ) -> Tuple[float, Dict]:
        """Estimate beta for a single asset."""
        # Remove NaN values
        valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
        asset_returns = asset_returns[valid_mask]
        market_returns = market_returns[valid_mask]
        
        n = len(asset_returns)
        
        if n < self.min_observations:
            return np.nan, {'error': 'insufficient_data', 'n': n}
        
        K = max(2, int(self.c_omega * np.sqrt(n)))
        
        # Use provided alpha, or current rolling alpha, or None (individual)
        alpha_to_use = universal_alpha if universal_alpha is not None else self._current_alpha
        
        if self.use_tarv:
            beta, info = calculate_realized_beta(
                asset_returns=asset_returns,
                market_returns=market_returns,
                K=K,
                p=2,
                use_tarv=True,
                universal_alpha=alpha_to_use
            )
            diagnostics = {
                'method': 'TARV',
                'K': K,
                'truncation_rate': info.get('var_truncation_rate', np.nan),
                'tail_alpha': info.get('var_alpha', np.nan),
                'universal_alpha_used': alpha_to_use
            }
        else:
            cov = np.sum(asset_returns * market_returns)
            var = np.sum(market_returns ** 2)
            beta = cov / var if var != 0 else np.nan
            diagnostics = {'method': 'Standard'}
        
        return beta, diagnostics
    
    def estimate_all_betas(
        self,
        assets_data: Dict[str, np.ndarray],
        market_returns: np.ndarray,
        universal_alpha: float = None
    ) -> Dict[str, Tuple[float, Dict]]:
        """Estimate betas for all assets."""
        results = {}
        
        # If using rolling alpha and no alpha provided, estimate from current window
        if self.use_rolling_alpha and universal_alpha is None and self.use_tarv:
            universal_alpha = self.estimate_rolling_alpha(assets_data, market_returns)
        
        for asset_id, asset_returns in assets_data.items():
            beta, diag = self.estimate_beta(asset_returns, market_returns, universal_alpha)
            results[asset_id] = (beta, diag)
            if self.verbose:
                print(f"  {asset_id}: β = {beta:.4f}")
        return results
    
    def estimate_all_betas_fast(
        self,
        assets_df: pd.DataFrame,
        market_returns: np.ndarray
    ) -> Dict[str, Tuple[float, Dict]]:
        """
        Fast vectorized beta estimation for all assets.
        Uses Standard RV only (no TARV) for speed.
        """
        results = {}
        
        # Convert to numpy for speed
        assets_matrix = assets_df.values  # (n_obs, n_assets)
        market = market_returns.reshape(-1, 1)  # (n_obs, 1)
        
        # Vectorized covariance and variance
        # cov(asset, market) = sum(asset * market)
        # var(market) = sum(market^2)
        cov_matrix = np.nansum(assets_matrix * market, axis=0)  # (n_assets,)
        var_market = np.nansum(market ** 2)
        
        # Betas = cov / var
        betas = cov_matrix / var_market if var_market != 0 else np.full(len(cov_matrix), np.nan)
        
        # Create results dict
        for i, col in enumerate(assets_df.columns):
            results[col] = (betas[i], {'method': 'Standard_Fast'})
        
        return results


# =============================================================================
# Portfolio Constructor
# =============================================================================

class PortfolioConstructor:
    """Construct beta-sorted portfolios."""
    
    def __init__(
        self,
        n_portfolios: int = 5,
        weighting: str = 'equal',
        long_short: bool = True
    ):
        self.n_portfolios = n_portfolios
        self.weighting = weighting
        self.long_short = long_short
    
    def _get_portfolio_names(self) -> List[str]:
        """Get portfolio names based on number of portfolios."""
        if self.n_portfolios == 3:
            return ["Low_Beta", "Mid_Beta", "High_Beta"]
        elif self.n_portfolios == 4:
            return ["Low_Beta", "MidLow_Beta", "MidHigh_Beta", "High_Beta"]
        elif self.n_portfolios == 5:
            return ["Low_Beta", "Q2", "Q3", "Q4", "High_Beta"]
        else:
            names = [f"P{i+1}" for i in range(self.n_portfolios)]
            names[0] = "Low_Beta"
            names[-1] = "High_Beta"
            return names
    
    def construct_portfolios(
        self,
        betas: Dict[str, float],
        timestamp: datetime
    ) -> Dict[str, PortfolioWeights]:
        """Construct beta-sorted portfolios."""
        # Remove NaN betas
        valid_betas = {k: v for k, v in betas.items() if not np.isnan(v)}
        n_assets = len(valid_betas)
        
        if n_assets < self.n_portfolios:
            warnings.warn(f"Only {n_assets} assets, need at least {self.n_portfolios}")
            return {}
        
        # Sort by beta
        sorted_assets = sorted(valid_betas.items(), key=lambda x: x[1])
        
        # Divide into portfolios
        assets_per_portfolio = n_assets // self.n_portfolios
        portfolios = {}
        portfolio_names = self._get_portfolio_names()
        
        for i in range(self.n_portfolios):
            start_idx = i * assets_per_portfolio
            end_idx = n_assets if i == self.n_portfolios - 1 else (i + 1) * assets_per_portfolio
            
            portfolio_assets = sorted_assets[start_idx:end_idx]
            
            # Equal weights
            weight = 1.0 / len(portfolio_assets)
            weights = {asset: weight for asset, _ in portfolio_assets}
            asset_betas = {asset: b for asset, b in portfolio_assets}
            
            portfolios[portfolio_names[i]] = PortfolioWeights(
                timestamp=timestamp,
                weights=weights,
                betas=asset_betas,
                portfolio_type=portfolio_names[i]
            )
        
        # Create BAB (Betting Against Beta) portfolio with beta leverage adjustment
        # Following Frazzini & Pedersen (2014):
        # BAB = (1/β_L) × Low_Beta - (1/β_H) × High_Beta
        if self.long_short and "Low_Beta" in portfolios and "High_Beta" in portfolios:
            # Calculate portfolio betas
            beta_L = np.mean(list(portfolios["Low_Beta"].betas.values()))
            beta_H = np.mean(list(portfolios["High_Beta"].betas.values()))
            
            # Avoid division by zero
            beta_L = max(beta_L, 0.1)
            beta_H = max(beta_H, 0.1)
            
            # Beta-adjusted weights: (1/β_L) for long, -(1/β_H) for short
            ls_weights = {}
            for asset, w in portfolios["Low_Beta"].weights.items():
                ls_weights[asset] = w / beta_L  # Long with leverage 1/β_L
            for asset, w in portfolios["High_Beta"].weights.items():
                ls_weights[asset] = -w / beta_H  # Short with leverage 1/β_H
            
            all_betas = {**portfolios["Low_Beta"].betas, **portfolios["High_Beta"].betas}
            
            portfolios["BAB"] = PortfolioWeights(
                timestamp=timestamp,
                weights=ls_weights,
                betas=all_betas,
                portfolio_type="long_short"
            )
        
        return portfolios


# =============================================================================
# Backtest Engine
# =============================================================================

class BacktestEngine:
    """Backtest beta-sorted portfolios."""
    
    def __init__(
        self,
        beta_estimator: BetaEstimator,
        portfolio_constructor: PortfolioConstructor,
        rebalance_frequency: str = 'monthly',
        transaction_cost: float = 0.001
    ):
        self.beta_estimator = beta_estimator
        self.portfolio_constructor = portfolio_constructor
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.portfolio_history: List[Dict[str, PortfolioWeights]] = []
        self.dates: List[datetime] = []
    
    def run_backtest(
        self,
        assets_prices: pd.DataFrame,
        market_prices: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """Run backtest."""
        print("🔍 Validating data...")
        
        # Data validation
        assets_prices = assets_prices.select_dtypes(include=['float', 'float64'])
        if isinstance(market_prices, pd.DataFrame):
            market_prices = market_prices.select_dtypes(include=['float', 'float64']).iloc[:, 0]
        market_prices = pd.to_numeric(market_prices, errors='coerce').dropna()
        
        if assets_prices.empty or market_prices.empty:
            raise ValueError("No valid numeric data found")
        
        print("✅ Data validation passed")
        
        # Align data
        common_idx = assets_prices.index.intersection(market_prices.index)
        assets_prices = assets_prices.loc[common_idx]
        market_prices = market_prices.loc[common_idx]
        
        # Calculate returns
        assets_returns = assets_prices.pct_change().dropna()
        market_returns = market_prices.pct_change().dropna()
        
        # Align returns
        common_idx = assets_returns.index.intersection(market_returns.index)
        assets_returns = assets_returns.loc[common_idx]
        market_returns = market_returns.loc[common_idx]
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(assets_returns.index)
        
        print(f"📊 Backtest: {len(rebalance_dates)} rebalance periods")
        print(f"   Assets: {len(assets_returns.columns)}")
        print(f"   Date range: {assets_returns.index[0]} to {assets_returns.index[-1]}")
        
        # Initialize portfolio returns
        portfolio_names = self.portfolio_constructor._get_portfolio_names() + ['BAB', 'Market']
        portfolio_returns = {name: [] for name in portfolio_names}
        
        # Rolling window backtest
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_rebal_date = rebalance_dates[i + 1]
            
            # Estimation window: use observation count, not time delta
            # Find index position of rebal_date
            rebal_idx = assets_returns.index.get_indexer([rebal_date], method='ffill')[0]
            if rebal_idx < 0:
                rebal_idx = assets_returns.index.searchsorted(rebal_date)
            
            # Get window by observation count
            window_size = self.beta_estimator.window_observations
            start_idx = max(0, rebal_idx - window_size + 1)
            
            est_assets = assets_returns.iloc[start_idx:rebal_idx + 1]
            est_market = market_returns.iloc[start_idx:rebal_idx + 1]
            
            if len(est_assets) < self.beta_estimator.min_observations:
                continue
            
            # Estimate betas
            assets_dict = {col: est_assets[col].values for col in est_assets.columns}
            beta_results = self.beta_estimator.estimate_all_betas(assets_dict, est_market.values)
            betas = {k: v[0] for k, v in beta_results.items()}
            
            # Construct portfolios
            portfolios = self.portfolio_constructor.construct_portfolios(betas, rebal_date)
            if not portfolios:
                continue
            
            self.portfolio_history.append(portfolios)
            
            # Holding period returns
            hold_mask = (assets_returns.index > rebal_date) & (assets_returns.index <= next_rebal_date)
            hold_assets = assets_returns.loc[hold_mask]
            hold_market = market_returns.loc[hold_mask]
            
            if len(hold_assets) == 0:
                continue
            
            # Calculate portfolio returns
            for port_name, port in portfolios.items():
                port_ret = sum(
                    weight * ((1 + hold_assets[asset]).prod() - 1)
                    for asset, weight in port.weights.items()
                    if asset in hold_assets.columns
                )
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
            dates = index.normalize().unique()
        elif self.rebalance_frequency == 'weekly':
            dates = index[index.dayofweek == 0].normalize().unique()
        elif self.rebalance_frequency == 'monthly':
            dates = index[index.is_month_start].normalize().unique()
        else:
            dates = index.normalize().unique()
        return list(dates)
    
    def calculate_performance(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 12
    ) -> PortfolioPerformance:
        """Calculate portfolio performance metrics."""
        returns = returns_df['return'].values
        
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = np.min((cumulative - running_max) / running_max)
        
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


# =============================================================================
# Output Functions
# =============================================================================

def print_performance_summary(results: Dict[str, pd.DataFrame], engine: BacktestEngine):
    """Print performance summary for all portfolios."""
    print("\n" + "=" * 70)
    print("📈 PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Determine periods per year
    freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods_per_year = freq_map.get(engine.rebalance_frequency, 12)
    
    for port_name, returns_df in results.items():
        if returns_df is None or len(returns_df) == 0:
            continue
        
        perf = engine.calculate_performance(returns_df, periods_per_year=periods_per_year)
        
        print(f"\n🔹 {port_name}")
        print(f"   Total Return:      {perf.total_return * 100:>8.2f}%")
        print(f"   Annualized Return: {perf.annualized_return * 100:>8.2f}%")
        print(f"   Volatility:        {perf.volatility * 100:>8.2f}%")
        print(f"   Sharpe Ratio:      {perf.sharpe_ratio:>8.2f}")
        print(f"   Max Drawdown:      {perf.max_drawdown * 100:>8.2f}%")
        if not np.isnan(perf.avg_beta):
            print(f"   Avg Beta:          {perf.avg_beta:>8.2f}")
        print(f"   # Rebalances:      {perf.n_rebalances:>8d}")
    
    # Low-Beta Anomaly Analysis
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


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_real_data_analysis(
    data_dir: str = "data",
    exchange: str = "binance",
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_tarv: bool = True,
    n_portfolios: int = 3,
    rebalance_frequency: str = 'weekly',
    window_days: int = 30,
    min_observations: int = 10000
) -> Dict[str, pd.DataFrame]:
    """
    Run Low-Beta Anomaly analysis on real data.
    
    Args:
        data_dir: Path to data directory
        exchange: 'binance' or 'upbit'
        symbols: List of symbols (None = all available)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_tarv: Use TARV for beta estimation
        n_portfolios: Number of beta-sorted portfolios (3, 4, or 5)
        rebalance_frequency: 'daily', 'weekly', or 'monthly'
        window_days: Rolling window for beta estimation
        min_observations: Minimum observations per asset
    
    Returns:
        Dict of portfolio results
    """
    from data_loader import prepare_data_for_analysis
    
    print("=" * 70)
    print("🔬 LOW-BETA ANOMALY ANALYSIS")
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
        return {}
    
    # Configuration
    minutes_per_day = 1440 if exchange in ['binance', 'upbit'] else 390
    method_name = "TARV" if use_tarv else "Standard RV"
    
    print(f"\n🔧 Configuration:")
    print(f"   Beta estimation: {method_name}")
    print(f"   Rolling window: {window_days} days")
    print(f"   Portfolios: {n_portfolios}")
    print(f"   Rebalancing: {rebalance_frequency}")
    
    # Initialize components
    beta_estimator = BetaEstimator(
        window_observations=minutes_per_day * window_days,
        min_observations=min_observations,
        use_tarv=use_tarv,
        c_omega=0.333
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
    print("\n🔄 Running backtest...")
    results = backtest_engine.run_backtest(
        assets_prices=assets_prices,
        market_prices=market_prices
    )
    
    # Print results
    print_performance_summary(results, backtest_engine)
    
    # Statistical test
    if 'BAB' in results and len(results['BAB']) > 0:
        bab_returns = results['BAB']['return']
        
        print("\n" + "-" * 70)
        print("📊 STATISTICAL SIGNIFICANCE TEST")
        print("-" * 70)
        
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(bab_returns, 0)
        
        print(f"   BAB Mean Return: {bab_returns.mean() * 100:.4f}%")
        print(f"   BAB Std Dev:     {bab_returns.std() * 100:.4f}%")
        print(f"   T-statistic:     {t_stat:.4f}")
        print(f"   P-value:         {p_value:.4f}")
        
        if p_value < 0.05:
            if bab_returns.mean() > 0:
                print("\n   ✅ SIGNIFICANT LOW-BETA ANOMALY (p < 0.05)")
            else:
                print("\n   ⚠️  SIGNIFICANT but REVERSED (p < 0.05)")
        else:
            print("\n   ❌ NOT SIGNIFICANT (p >= 0.05)")
    
    print("\n" + "=" * 70)
    print("✅ Analysis completed!")
    print("=" * 70)
    
    return results
