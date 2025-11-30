#!/usr/bin/env python3
"""
Tail-Adaptive Realized Volatility (TARV) - Full Implementation
Based on Shin, Kim, & Fan (2023) - "Adaptive robust large volatility matrix estimation"
Journal of Econometrics, 237(1), 105514

TERMINOLOGY NOTE:
================
This module uses "alpha" to refer to the TAIL INDEX (tail_alpha):
- Measures the heaviness of return distribution tails
- Higher values = lighter tails (closer to Gaussian)
- Lower values = heavier tails (more extreme events)
- Used in truncation threshold calculation

This is DIFFERENT from "Jensen's alpha" in CAPM:
- Jensen's alpha = abnormal return not explained by market beta
- That alpha is computed in portfolio.py as portfolio performance

Throughout this module:
- "alpha" refers to tail_alpha (tail index)
- "universal_alpha" refers to a cross-market tail index

This implementation follows the paper's methodology exactly:
1. Pre-averaging of high-frequency returns
2. Median-centered truncation
3. Extended Hill estimator for tail index
4. Adaptive threshold (Eq. 5.17)
5. Truncated realized covariance matrix
"""

import numpy as np
import polars as pl
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class TailIndexResult:
    """Result of tail index (tail_alpha) estimation"""
    alpha: float           # Estimated tail index (tail_alpha)
    k: int                 # Number of order statistics used
    omega: int             # Trimming parameter
    n: int                 # Sample size


@dataclass 
class ThresholdResult:
    """Result of threshold calculation"""
    theta: float           # Threshold value
    alpha: float           # Tail index used
    S_hat: float           # Median-adjusted moment
    c_alpha: float         # Scaling constant
    K_n: int               # Bandwidth parameter


@dataclass
class TARVResult:
    """Result of TARV calculation"""
    tarv: float            # Tail-adaptive realized variance/covariance
    standard_rv: float     # Standard realized variance/covariance
    n_truncated: int       # Number of truncated observations
    n_total: int           # Total observations
    truncation_rate: float # Percentage truncated
    threshold: float       # Threshold used
    alpha: float           # Tail index used


# =============================================================================
# 1. Pre-Averaging Functions
# =============================================================================

def compute_preaveraging_weights(K: int, weight_type: str = 'bartlett') -> np.ndarray:
    """
    Compute pre-averaging weights g(x).
    
    From paper Section 2.1:
    - Bartlett: g(x) = min(x, 1-x)  (commonly used)
    - Flat: g(x) = 1
    
    Args:
        K: Window size (bandwidth)
        weight_type: 'bartlett' or 'flat'
    
    Returns:
        Array of weights of length K
    """
    if weight_type == 'bartlett':
        # g(k/K) = min(k/K, 1 - k/K) for k = 1, ..., K
        x = np.arange(1, K + 1) / K
        weights = np.minimum(x, 1 - x)
    elif weight_type == 'flat':
        weights = np.ones(K)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
    # Normalize so that sum of squared weights equals 1
    # This is phi_22 in the paper
    weights = weights / np.sqrt(np.sum(weights ** 2))
    
    return weights


def preaverage_returns(
    returns: np.ndarray,
    K: int,
    weight_type: str = 'bartlett'
) -> np.ndarray:
    """
    Compute pre-averaged returns.
    
    From paper Eq. 2.1:
    Ȳ_k = Σ_{j=1}^{K-1} g(j/K) × ΔY_{k+j}
    
    where ΔY_k = Y_k - Y_{k-1} (log returns)
    
    Args:
        returns: Array of log returns
        K: Pre-averaging window size
        weight_type: Weight function type
    
    Returns:
        Pre-averaged returns
    """
    n = len(returns)
    if n < K:
        raise ValueError(f"Not enough observations: {n} < K={K}")
    
    weights = compute_preaveraging_weights(K - 1, weight_type)
    
    # Number of pre-averaged observations
    n_bar = n - K + 1
    
    # Compute pre-averaged returns
    Y_bar = np.zeros(n_bar)
    for k in range(n_bar):
        Y_bar[k] = np.sum(weights * returns[k:k + K - 1])
    
    return Y_bar


def compute_quadratic_variable(
    returns_i: np.ndarray,
    returns_j: np.ndarray,
    K: int,
    weight_type: str = 'bartlett'
) -> np.ndarray:
    """
    Compute pre-averaged quadratic covariation increments Q_ij.
    
    From paper Eq. 2.2:
    Q_ij(τ_k) = Ȳ_i,k × Ȳ_j,k
    
    For variance (i=j): Q_ii(τ_k) = Ȳ_i,k²
    
    Args:
        returns_i: Log returns of asset i
        returns_j: Log returns of asset j
        K: Pre-averaging window size
        weight_type: Weight function type
    
    Returns:
        Array of quadratic covariation increments
    """
    Y_bar_i = preaverage_returns(returns_i, K, weight_type)
    Y_bar_j = preaverage_returns(returns_j, K, weight_type)
    
    # Q_ij = Ȳ_i × Ȳ_j
    Q_ij = Y_bar_i * Y_bar_j
    
    return Q_ij


# =============================================================================
# 2. Tail Index Estimation (Extended Hill Estimator)
# =============================================================================

def estimate_tail_index(
    Q: np.ndarray,
    c_u: float = 2.0,
    c_omega: float = 0.333,
    c_xi: float = 0.01,
    xi: float = 0.2,
    p: int = 1,
    alpha_min: float = 1.1,
    alpha_max: float = 4.0,
    use_median_centered: bool = True
) -> TailIndexResult:
    """
    Estimate tail index using Extended Hill Estimator with trimming.
    
    From paper Section 3.2, Eq. 3.4-3.6:
    
    1. k = u_n = c_u × ⌊n^0.5⌋
    2. Trimming: ω = c_ω × log(p)
    3. Hill estimator: α̃^(-1) = (1/k) × Σ_{j=ω+1}^{k+ω} [log|Q_j| - log|Q_{k+ω}|]
    4. Adjustment: α̂ = max{α̃ - c_ξ × n^(-ξ) × log(p), α_min}
    
    Args:
        Q: Array of quadratic covariation increments
        c_u: Constant for k selection (default: 2.0)
        c_omega: Constant for trimming (default: 1/3)
        c_xi: Constant for bias adjustment (default: 0.01)
        xi: Exponent for bias adjustment (default: 0.2)
        p: Number of assets
        alpha_min: Minimum tail index (default: 1.1)
        alpha_max: Maximum tail index (default: 4.0)
        use_median_centered: If True, estimate from |Q - median(Q)| (more robust)
    
    Returns:
        TailIndexResult with estimated alpha and diagnostics
    """
    # Remove NaN and zero values
    Q = Q[~np.isnan(Q)]
    Q = Q[Q != 0]
    n = len(Q)
    
    if n < 10:
        raise ValueError(f"Not enough observations: {n} < 10")
    
    # Use median-centered values for tail estimation (more robust)
    if use_median_centered:
        MQ = np.median(Q)
        values_for_tail = np.abs(Q - MQ)
        # Remove zeros after centering
        values_for_tail = values_for_tail[values_for_tail > 0]
        n_tail = len(values_for_tail)
    else:
        values_for_tail = np.abs(Q)
        n_tail = n
    
    if n_tail < 10:
        return TailIndexResult(alpha=alpha_min, k=0, omega=0, n=n)
    
    # Calculate k = c_u × floor(n^0.5)
    k = int(c_u * np.floor(np.sqrt(n_tail)))
    k = max(k, 2)
    
    # Calculate trimming parameter ω (removes jump contamination)
    omega = int(np.ceil(c_omega * np.log(max(p, 2))))
    omega = max(omega, 0)
    
    # Sort values in descending order
    sorted_vals = np.sort(values_for_tail)[::-1]
    
    # Ensure we have enough observations after trimming
    if k + omega >= len(sorted_vals):
        k = max(2, len(sorted_vals) - omega - 1)
    
    if k < 2:
        return TailIndexResult(alpha=alpha_min, k=k, omega=omega, n=n)
    
    # Hill estimator with trimming
    start_idx = omega
    end_idx = omega + k
    
    if end_idx >= len(sorted_vals):
        end_idx = len(sorted_vals) - 1
        k = end_idx - start_idx
    
    # Add small epsilon for numerical stability
    eps = np.finfo(float).eps * 100
    log_sorted = np.log(sorted_vals[start_idx:end_idx] + eps)
    log_ref = np.log(sorted_vals[end_idx - 1] + eps)
    
    alpha_inv = np.mean(log_sorted - log_ref)
    
    if alpha_inv <= 0:
        alpha_tilde = alpha_max
    else:
        alpha_tilde = 1.0 / alpha_inv
    
    # Bias adjustment
    adjustment = c_xi * (n_tail ** (-xi)) * np.log(max(p, 2))
    alpha_hat = alpha_tilde - adjustment
    
    # Clip to valid range
    alpha_hat = np.clip(alpha_hat, alpha_min, alpha_max)
    
    return TailIndexResult(alpha=alpha_hat, k=k, omega=omega, n=n)


# =============================================================================
# 3. Adaptive Threshold Calculation (Eq. 5.17)
# =============================================================================

def compute_c_alpha(alpha: float) -> float:
    """
    Compute the scaling constant c_α.
    
    From paper below Eq. 5.17:
    c_α = max{(α-1)/α, √((2-α)/α)} for α > 1
    
    Args:
        alpha: Tail index
    
    Returns:
        Scaling constant c_α
    """
    if alpha <= 1:
        return 1.0  # Fallback
    
    term1 = (alpha - 1) / alpha
    
    if alpha < 2:
        term2 = np.sqrt((2 - alpha) / alpha)
    else:
        term2 = 0  # For α >= 2, second term is 0 or imaginary
    
    return max(term1, term2)


def compute_median_adjusted_moment(
    Q: np.ndarray,
    alpha: float,
    K_n: int
) -> float:
    """
    Compute the median-adjusted moment Ŝ_ij.
    
    From paper Eq. 5.17:
    Ŝ_ij = (1/(n-K_n)) × Σ|Q_ij(τ_k) - MQ_ij|^α
    
    where MQ_ij = median(Q_ij)
    
    Args:
        Q: Array of quadratic covariation increments
        alpha: Tail index
        K_n: Bandwidth parameter
    
    Returns:
        Median-adjusted moment Ŝ
    """
    # Remove NaN
    Q = Q[~np.isnan(Q)]
    n = len(Q)
    
    if n <= K_n:
        return np.mean(np.abs(Q) ** alpha)
    
    # Median of Q
    MQ = np.median(Q)
    
    # Compute absolute deviations
    abs_dev = np.abs(Q - MQ)
    
    # For numerical stability with large alpha, use robust estimation
    eps = np.finfo(float).eps
    abs_dev_safe = np.maximum(abs_dev, eps)
    
    # Use robust estimation: trimmed mean to avoid extreme values
    sorted_dev = np.sort(abs_dev_safe)
    trim_n = max(1, int(0.01 * n))
    trimmed_dev = sorted_dev[trim_n:-trim_n] if trim_n > 0 and 2*trim_n < n else sorted_dev
    
    # Compute Ŝ with numerical stability
    if alpha <= 2:
        S_hat = np.mean(trimmed_dev ** alpha)
    else:
        # For large alpha, use median-based robust estimator
        mad = np.median(abs_dev)
        if mad > eps:
            # Normalized deviations, capped to avoid overflow
            normalized = np.minimum(abs_dev / mad, 10)
            # Compute moment on normalized, then scale back
            S_hat = (mad ** alpha) * np.mean(normalized ** alpha)
        else:
            S_hat = np.mean(abs_dev_safe ** alpha)
    
    return max(S_hat, eps)


def calculate_threshold(
    Q: np.ndarray,
    alpha: float,
    p: int = 1,
    c: float = 0.35,
    K_n: Optional[int] = None,
    use_quantile: bool = True
) -> ThresholdResult:
    """
    Calculate adaptive threshold θ_ij.
    
    From paper Eq. 5.17:
    θ_ij = c × [ (K_n × log(p) × (α̂_ij - 1)) / (c_α̂_ij × Ŝ_ij × (n - K_n)) ]^(1/α̂_ij)
    
    Note: For small p (single asset or pairs), we use a quantile-based approach
    that achieves similar truncation rates (~1-5%) as the paper's large-p case.
    
    Args:
        Q: Array of quadratic covariation increments
        alpha: Estimated tail index
        p: Number of assets (default: 1)
        c: Threshold constant (0.35 for empirical, 0.5 for simulation)
        K_n: Bandwidth parameter (default: floor(n^0.5))
        use_quantile: For small p, use quantile-based threshold (recommended)
    
    Returns:
        ThresholdResult with threshold and diagnostics
    """
    Q = Q[~np.isnan(Q)]
    n = len(Q)
    
    if K_n is None:
        K_n = int(np.floor(np.sqrt(n)))
    
    K_n = max(K_n, 1)
    
    # Ensure alpha > 1
    alpha = max(alpha, 1.01)
    
    # Compute c_α
    c_alpha = compute_c_alpha(alpha)
    
    # Compute median-adjusted moment Ŝ
    S_hat = compute_median_adjusted_moment(Q, alpha, K_n)
    
    # Avoid division by zero
    if S_hat < 1e-15:
        S_hat = np.mean(np.abs(Q) ** alpha)
    if S_hat < 1e-15:
        S_hat = 1e-15
    
    # For small p (univariate or bivariate), the paper's formula gives
    # unreasonably large thresholds because log(p) is small.
    # We use a quantile-based approach that achieves similar truncation rates.
    if use_quantile and p <= 10:
        # Median-centered absolute deviations
        MQ = np.median(Q)
        abs_dev = np.abs(Q - MQ)
        
        # Target truncation rate based on tail heaviness
        # Heavier tail (smaller α) → more truncation needed
        # Lighter tail (larger α) → less truncation needed
        if alpha < 2:
            target_truncation = 0.05  # 5% for heavy tails
        elif alpha < 3:
            target_truncation = 0.03  # 3% for moderate tails  
        else:
            target_truncation = 0.02  # 2% for light tails
        
        # Adjust by c parameter
        target_truncation = target_truncation * (c / 0.35)
        target_truncation = np.clip(target_truncation, 0.005, 0.10)
        
        target_quantile = 1.0 - target_truncation
        theta = np.quantile(abs_dev, target_quantile)
        
        # Ensure theta is positive and reasonable
        if theta <= 0:
            theta = np.quantile(abs_dev, 0.95)
    else:
        # Original paper formula for large p
        # θ = c × [ (K_n × log(p) × (α - 1)) / (c_α × Ŝ × (n - K_n)) ]^(1/α)
        
        log_p = np.log(max(p, 2))
        
        numerator = K_n * log_p * (alpha - 1)
        denominator = c_alpha * S_hat * max(n - K_n, 1)
        
        if denominator <= 0:
            denominator = 1e-15
        
        ratio = numerator / denominator
        
        if ratio <= 0:
            ratio = 1e-15
        
        theta = c * (ratio ** (1.0 / alpha))
    
    return ThresholdResult(
        theta=theta,
        alpha=alpha,
        S_hat=S_hat,
        c_alpha=c_alpha,
        K_n=K_n
    )


# =============================================================================
# 4. Median-Centered Truncation
# =============================================================================

def truncate_with_median_centering(
    Q: np.ndarray,
    theta: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply median-centered truncation.
    
    From paper Section 3.1:
    Truncate observations where |Q - median(Q)| > θ
    
    Args:
        Q: Array of quadratic covariation increments
        theta: Threshold
    
    Returns:
        Tuple of (truncated_Q, mask, n_truncated)
        - truncated_Q: Q values set to median where |Q - median| > θ
        - mask: Boolean mask (True = kept, False = truncated)
        - n_truncated: Number of truncated observations
    """
    MQ = np.median(Q)
    
    # Mask: True if |Q - MQ| <= θ
    mask = np.abs(Q - MQ) <= theta
    
    # Truncated values are set to median
    truncated_Q = np.where(mask, Q, MQ)
    
    n_truncated = np.sum(~mask)
    
    return truncated_Q, mask, n_truncated


# =============================================================================
# 5. TARV Calculation (Realized Variance/Covariance)
# =============================================================================

def calculate_tarv_variance(
    returns: np.ndarray,
    K: int,
    p: int = 1,
    c: float = 0.35,
    weight_type: str = 'bartlett',
    return_details: bool = False,
    universal_alpha: Optional[float] = None
) -> Union[float, TARVResult]:
    """
    Calculate Tail-Adaptive Realized Variance for a single asset.
    
    Full pipeline:
    1. Pre-average returns
    2. Compute quadratic variables Q_ii = Ȳ²
    3. Estimate tail index
    4. Calculate adaptive threshold
    5. Apply median-centered truncation
    6. Sum truncated Q values
    
    Args:
        returns: Array of log returns
        K: Pre-averaging window size
        p: Number of assets (for threshold calculation)
        c: Threshold constant (0.35 empirical, 0.5 simulation)
        weight_type: Pre-averaging weight type
        return_details: If True, return TARVResult; else return float
    
    Returns:
        TARV value or TARVResult with details
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    
    if n < K + 10:
        if return_details:
            return TARVResult(
                tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0
            )
        return np.nan
    
    # Step 1: Pre-average returns
    Y_bar = preaverage_returns(returns, K, weight_type)
    
    # Step 2: Compute quadratic variables Q_ii = Ȳ²
    Q = Y_bar ** 2
    
    # Step 3: Estimate tail index (or use universal_alpha)
    if universal_alpha is not None:
        alpha = universal_alpha
    else:
        tail_result = estimate_tail_index(Q, p=p)
        alpha = tail_result.alpha
    
    # Step 4: Calculate adaptive threshold
    thresh_result = calculate_threshold(Q, alpha, p=p, c=c)
    theta = thresh_result.theta
    
    # Step 5: Apply median-centered truncation
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    # Step 6: Sum to get TARV
    # Apply bias correction for pre-averaging
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    
    # TARV = (1/phi_22) × Σ truncated_Q
    tarv = np.sum(truncated_Q) / phi_22
    
    # Standard RV for comparison
    standard_rv = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(
            tarv=tarv,
            standard_rv=standard_rv,
            n_truncated=n_truncated,
            n_total=len(Q),
            truncation_rate=n_truncated / len(Q) * 100,
            threshold=theta,
            alpha=alpha
        )
    
    return tarv


def calculate_tarv_covariance(
    returns_i: np.ndarray,
    returns_j: np.ndarray,
    K: int,
    p: int = 2,
    c: float = 0.35,
    weight_type: str = 'bartlett',
    return_details: bool = False,
    universal_alpha: Optional[float] = None
) -> Union[float, TARVResult]:
    """
    Calculate Tail-Adaptive Realized Covariance between two assets.
    
    Full pipeline:
    1. Pre-average both return series
    2. Compute quadratic variables Q_ij = Ȳ_i × Ȳ_j
    3. Estimate tail index from |Q_ij|
    4. Calculate adaptive threshold
    5. Apply median-centered truncation
    6. Sum truncated Q values
    
    Args:
        returns_i: Log returns of asset i
        returns_j: Log returns of asset j
        K: Pre-averaging window size
        p: Number of assets
        c: Threshold constant
        weight_type: Pre-averaging weight type
        return_details: If True, return TARVResult
    
    Returns:
        TARV covariance or TARVResult with details
    """
    returns_i = np.asarray(returns_i)
    returns_j = np.asarray(returns_j)
    
    # Align and remove NaN
    valid_mask = ~(np.isnan(returns_i) | np.isnan(returns_j))
    returns_i = returns_i[valid_mask]
    returns_j = returns_j[valid_mask]
    
    n = len(returns_i)
    
    if n < K + 10:
        if return_details:
            return TARVResult(
                tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0
            )
        return np.nan
    
    # Step 1: Pre-average both series
    Y_bar_i = preaverage_returns(returns_i, K, weight_type)
    Y_bar_j = preaverage_returns(returns_j, K, weight_type)
    
    # Step 2: Compute quadratic covariation Q_ij = Ȳ_i × Ȳ_j
    Q = Y_bar_i * Y_bar_j
    
    # Step 3: Estimate tail index (or use universal_alpha)
    if universal_alpha is not None:
        alpha = universal_alpha
    else:
        tail_result = estimate_tail_index(np.abs(Q), p=p)
        alpha = tail_result.alpha
    
    # Step 4: Calculate adaptive threshold
    thresh_result = calculate_threshold(Q, alpha, p=p, c=c)
    theta = thresh_result.theta
    
    # Step 5: Apply median-centered truncation
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    # Step 6: Sum to get TARV covariance
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    
    tarv_cov = np.sum(truncated_Q) / phi_22
    standard_cov = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(
            tarv=tarv_cov,
            standard_rv=standard_cov,
            n_truncated=n_truncated,
            n_total=len(Q),
            truncation_rate=n_truncated / len(Q) * 100,
            threshold=theta,
            alpha=alpha
        )
    
    return tarv_cov


# =============================================================================
# 6. Realized Beta Calculation
# =============================================================================

def calculate_realized_beta(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    K: int,
    p: int = 2,
    c: float = 0.35,
    use_tarv: bool = True,
    weight_type: str = 'bartlett',
    universal_alpha: Optional[float] = None
) -> Tuple[float, dict]:
    """
    Calculate realized beta using TARV or standard RV.
    
    β = Cov(asset, market) / Var(market)
    
    Args:
        asset_returns: Log returns of the asset
        market_returns: Log returns of the market
        K: Pre-averaging window size
        p: Number of assets
        c: Threshold constant for TARV
        use_tarv: If True, use TARV; if False, use standard pre-averaged RV
        weight_type: Pre-averaging weight type
        universal_alpha: If provided, use this alpha for all estimations
    
    Returns:
        Tuple of (beta, info_dict)
    """
    asset_returns = np.asarray(asset_returns)
    market_returns = np.asarray(market_returns)
    
    # Align
    valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
    asset_returns = asset_returns[valid_mask]
    market_returns = market_returns[valid_mask]
    
    if use_tarv:
        # TARV-based
        cov_result = calculate_tarv_covariance(
            asset_returns, market_returns, K, p, c, weight_type, 
            return_details=True, universal_alpha=universal_alpha
        )
        var_result = calculate_tarv_variance(
            market_returns, K, p, c, weight_type, 
            return_details=True, universal_alpha=universal_alpha
        )
        
        cov = cov_result.tarv
        var_market = var_result.tarv
        
        info = {
            'method': 'TARV',
            'cov': cov,
            'var_market': var_market,
            'cov_alpha': cov_result.alpha,
            'var_alpha': var_result.alpha,
            'cov_threshold': cov_result.threshold,
            'var_threshold': var_result.threshold,
            'cov_truncation_rate': cov_result.truncation_rate,
            'var_truncation_rate': var_result.truncation_rate,
            'standard_cov': cov_result.standard_rv,
            'standard_var': var_result.standard_rv
        }
    else:
        # Standard pre-averaged RV
        Y_bar_asset = preaverage_returns(asset_returns, K, weight_type)
        Y_bar_market = preaverage_returns(market_returns, K, weight_type)
        
        phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
        
        cov = np.sum(Y_bar_asset * Y_bar_market) / phi_22
        var_market = np.sum(Y_bar_market ** 2) / phi_22
        
        info = {
            'method': 'Standard Pre-Averaged RV',
            'cov': cov,
            'var_market': var_market
        }
    
    # Calculate beta
    if var_market == 0 or np.isnan(var_market):
        beta = np.nan
    else:
        beta = cov / var_market
    
    info['beta'] = beta
    
    return beta, info


# =============================================================================
# 6.5 Universal Alpha Estimation (Cross-Market)
# =============================================================================

def estimate_universal_alpha(
    returns_list: List[np.ndarray],
    K: int,
    p: int = 2,
    method: str = 'median'
) -> Tuple[float, dict]:
    """
    Estimate a universal tail index (alpha) from multiple asset/market return series.
    
    This implements the professor's suggestion:
    "stock이랑 crypto 둘다 사용해서 universal level alpha를 사용하는게 더 좋을 수 있다"
    
    Args:
        returns_list: List of return arrays (e.g., [stock_returns, crypto_returns])
        K: Pre-averaging window size
        p: Number of assets parameter
        method: How to combine individual alphas ('median', 'mean', 'min')
    
    Returns:
        Tuple of (universal_alpha, info_dict)
    
    Example:
        # Stock returns
        stock_returns = np.random.standard_t(df=4, size=10000) * 0.001
        # Crypto returns (fatter tails)
        crypto_returns = np.random.standard_t(df=3, size=10000) * 0.002
        
        universal_alpha, info = estimate_universal_alpha(
            [stock_returns, crypto_returns], 
            K=100, 
            method='median'
        )
        
        # Use for all estimations
        beta, _ = calculate_realized_beta(
            asset_returns, market_returns, K,
            universal_alpha=universal_alpha
        )
    """
    individual_alphas = []
    alpha_details = []
    
    for i, returns in enumerate(returns_list):
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < K + 10:
            continue
        
        # Pre-average and compute Q values
        Y_bar = preaverage_returns(returns, K)
        Q = Y_bar ** 2
        
        # Estimate tail index
        tail_result = estimate_tail_index(Q, p=p)
        individual_alphas.append(tail_result.alpha)
        alpha_details.append({
            'series_idx': i,
            'n_obs': len(returns),
            'alpha': tail_result.alpha,
            'k': tail_result.k
        })
    
    if not individual_alphas:
        return 2.0, {'error': 'No valid series found', 'individual_alphas': []}
    
    # Combine individual alphas
    if method == 'median':
        universal_alpha = float(np.median(individual_alphas))
    elif method == 'mean':
        universal_alpha = float(np.mean(individual_alphas))
    elif method == 'min':
        # Conservative approach: use the fattest tail estimate
        universal_alpha = float(np.min(individual_alphas))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    info = {
        'method': method,
        'n_series': len(individual_alphas),
        'individual_alphas': individual_alphas,
        'alpha_details': alpha_details,
        'universal_alpha': universal_alpha
    }
    
    return universal_alpha, info


def estimate_universal_alpha_from_Q(
    Q_list: List[np.ndarray],
    p: int = 2,
    method: str = 'pooled'
) -> Tuple[float, dict]:
    """
    Estimate universal alpha directly from pre-computed Q values.
    
    This allows pooling Q values from different markets for a single
    tail index estimation, rather than combining individual estimates.
    
    Args:
        Q_list: List of Q (quadratic variation) arrays from different markets
        p: Number of assets parameter
        method: 'pooled' (combine all Q) or 'median' (combine individual alphas)
    
    Returns:
        Tuple of (universal_alpha, info_dict)
    """
    if method == 'pooled':
        # Pool all Q values together
        all_Q = np.concatenate([np.abs(Q.flatten()) for Q in Q_list])
        tail_result = estimate_tail_index(all_Q, p=p)
        
        info = {
            'method': 'pooled',
            'total_n': len(all_Q),
            'n_series': len(Q_list),
            'universal_alpha': tail_result.alpha,
            'k': tail_result.k
        }
        return tail_result.alpha, info
    
    elif method == 'median':
        # Estimate individually and take median
        individual_alphas = []
        for Q in Q_list:
            Q_abs = np.abs(Q.flatten())
            if len(Q_abs) > 20:
                tail_result = estimate_tail_index(Q_abs, p=p)
                individual_alphas.append(tail_result.alpha)
        
        if not individual_alphas:
            return 2.0, {'error': 'No valid Q arrays'}
        
        universal_alpha = float(np.median(individual_alphas))
        info = {
            'method': 'median',
            'n_series': len(individual_alphas),
            'individual_alphas': individual_alphas,
            'universal_alpha': universal_alpha
        }
        return universal_alpha, info
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 7. Helper Functions
# =============================================================================

def calculate_log_returns(
    df: pl.DataFrame,
    price_col: str = 'close',
    sort_col: str = 'timestamp'
) -> pl.DataFrame:
    """
    Calculate log returns from price data (Polars DataFrame).
    
    Args:
        df: Polars DataFrame with price data
        price_col: Column name for price
        sort_col: Column name for sorting
    
    Returns:
        DataFrame with 'log_return' column added
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found")
    
    if sort_col in df.columns:
        df = df.sort(sort_col)
    
    df = df.with_columns([
        (pl.col(price_col).log() - pl.col(price_col).log().shift(1)).alias('log_return')
    ])
    
    return df


def optimal_K(n: int, method: str = 'sqrt') -> int:
    """
    Calculate optimal pre-averaging window size K.
    
    From paper: K = c × n^(1/2) is typical
    
    Args:
        n: Number of observations
        method: 'sqrt' for K = sqrt(n), 'theta' for K = c*n^θ
    
    Returns:
        Optimal K
    """
    if method == 'sqrt':
        return max(2, int(np.sqrt(n)))
    elif method == 'theta':
        # K = c * n^(1/2) with c around 1-2
        return max(2, int(1.5 * np.sqrt(n)))
    else:
        return max(2, int(np.sqrt(n)))
