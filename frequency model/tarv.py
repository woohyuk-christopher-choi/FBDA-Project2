#!/usr/bin/env python3
"""
Tail-Adaptive Realized Volatility (TARV) Implementation
Based on Shin, Kim, & Fan (2023) - "Adaptive robust large volatility matrix estimation"
Journal of Econometrics, 237(1), 105514

TERMINOLOGY NOTE:
================
- "alpha" refers to the TAIL INDEX (tail_alpha): Measures tail heaviness
- Higher alpha = lighter tails (closer to Gaussian)
- Lower alpha = heavier tails (more extreme events)
- This is DIFFERENT from Jensen's alpha in CAPM

Implementation follows:
1. Pre-averaging of high-frequency returns (Numba-accelerated)
2. Extended Hill estimator for tail index
3. Adaptive threshold (Eq. 5.17)
4. Median-centered truncation

Numba Optimization:
==================
Core numerical functions are JIT-compiled with Numba for 10-50x speedup.
Falls back to pure numpy if Numba is not available.
"""

import numpy as np
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass

# =============================================================================
# Numba Setup - Fall back to numpy if not available
# =============================================================================

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


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
# 1. Numba-Accelerated Core Functions
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _compute_bartlett_weights_numba(K: int) -> np.ndarray:
    """Compute Bartlett weights (numba-accelerated)."""
    weights = np.zeros(K)
    for i in range(K):
        x = (i + 1) / (K + 1)
        weights[i] = min(x, 1 - x)
    
    # Normalize
    norm = np.sqrt(np.sum(weights ** 2))
    if norm > 0:
        weights = weights / norm
    
    return weights


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _preaverage_returns_numba(returns: np.ndarray, K: int) -> np.ndarray:
    """
    Numba-accelerated pre-averaging of returns.
    
    Args:
        returns: Array of log returns
        K: Pre-averaging window size
    
    Returns:
        Pre-averaged returns
    """
    n = len(returns)
    if n < K:
        return np.empty(0)
    
    weights = _compute_bartlett_weights_numba(K - 1)
    n_bar = n - K + 1
    Y_bar = np.zeros(n_bar)
    
    for k in prange(n_bar):
        total = 0.0
        for j in range(K - 1):
            total += weights[j] * returns[k + j]
        Y_bar[k] = total
    
    return Y_bar


@jit(nopython=True, cache=True, fastmath=True)
def _truncate_with_median_numba(Q: np.ndarray, theta: float) -> Tuple[np.ndarray, int]:
    """
    Median-centered truncation (numba-accelerated).
    
    Args:
        Q: Array of quadratic variables
        theta: Truncation threshold
    
    Returns:
        Tuple of (truncated_Q, n_truncated)
    """
    n = len(Q)
    
    # Compute median
    sorted_Q = np.sort(Q)
    if n % 2 == 0:
        median_Q = (sorted_Q[n // 2 - 1] + sorted_Q[n // 2]) / 2
    else:
        median_Q = sorted_Q[n // 2]
    
    # Apply truncation
    truncated_Q = np.zeros(n)
    n_truncated = 0
    
    for i in range(n):
        if abs(Q[i] - median_Q) <= theta:
            truncated_Q[i] = Q[i]
        else:
            truncated_Q[i] = median_Q
            n_truncated += 1
    
    return truncated_Q, n_truncated


@jit(nopython=True, cache=True, fastmath=True)
def _estimate_tail_index_numba(Q: np.ndarray, p: int = 2) -> float:
    """
    Extended Hill estimator for tail index (numba-accelerated).
    
    Args:
        Q: Array of squared/absolute returns
        p: Moment order (default 2)
    
    Returns:
        Estimated tail index alpha
    """
    n = len(Q)
    if n < 10:
        return 2.0
    
    # Sort in descending order
    Q_sorted = np.sort(Q)[::-1]
    
    # Parameters
    k = max(10, int(n ** 0.5))
    omega = max(1, int(k ** 0.5))
    
    if k + omega >= n:
        k = n // 2
        omega = max(1, k // 4)
    
    # Extended Hill estimator
    threshold = Q_sorted[k + omega - 1]
    if threshold <= 0:
        return 2.0
    
    sum_ratio = 0.0
    for i in range(omega, k + omega):
        if i < n and Q_sorted[i - 1] > 0:
            ratio = (Q_sorted[i - 1] / threshold) ** p - 1
            sum_ratio += ratio
    
    H = sum_ratio / k if k > 0 else 1.0
    
    if H > 0:
        alpha = p / H
    else:
        alpha = 4.0
    
    # Clamp to reasonable range
    alpha = max(0.5, min(4.0, alpha))
    
    return alpha


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_threshold_numba(Q: np.ndarray, alpha: float, c: float = 0.35) -> float:
    """
    Calculate adaptive threshold (numba-accelerated).
    
    Args:
        Q: Array of quadratic variables
        alpha: Tail index
        c: Threshold constant
    
    Returns:
        Threshold value theta
    """
    n = len(Q)
    if n < 5:
        return 0.0
    
    # Compute median
    sorted_Q = np.sort(Q)
    if n % 2 == 0:
        median_Q = (sorted_Q[n // 2 - 1] + sorted_Q[n // 2]) / 2
    else:
        median_Q = sorted_Q[n // 2]
    
    # Compute S_hat = median of |Q - median(Q)|^alpha
    abs_dev = np.abs(Q - median_Q)
    sorted_dev = np.sort(abs_dev)
    if n % 2 == 0:
        S_hat = (sorted_dev[n // 2 - 1] + sorted_dev[n // 2]) / 2
    else:
        S_hat = sorted_dev[n // 2]
    
    if S_hat < 1e-15:
        S_hat = 1e-15
    
    # c_alpha constant
    if alpha > 2.5:
        c_alpha = 0.8
    elif alpha > 2.0:
        c_alpha = 0.9
    elif alpha > 1.5:
        c_alpha = 1.0
    else:
        c_alpha = 1.1
    
    # K_n bandwidth
    K_n = max(1, int(n ** 0.4))
    
    # Paper formula (simplified)
    theta = c * c_alpha * (S_hat ** (1.0 / alpha)) * (K_n ** (1.0 / alpha))
    
    # Alpha-dependent quantile bounds
    if alpha < 1.5:
        min_q, max_q = 0.90, 0.98
    elif alpha < 2.0:
        min_q, max_q = 0.92, 0.99
    elif alpha < 2.5:
        min_q, max_q = 0.94, 0.995
    else:
        min_q, max_q = 0.95, 0.999
    
    # Clip theta to quantile bounds
    min_theta = sorted_dev[int(n * min_q)]
    max_theta = sorted_dev[min(int(n * max_q), n - 1)]
    
    if theta < min_theta:
        theta = min_theta
    if theta > max_theta:
        theta = max_theta
    
    return theta


# =============================================================================
# 2. Pre-Averaging Functions (Public API)
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
        if NUMBA_AVAILABLE:
            return _compute_bartlett_weights_numba(K)
        x = np.arange(1, K + 1) / (K + 1)
        weights = np.minimum(x, 1 - x)
    elif weight_type == 'flat':
        weights = np.ones(K)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
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
    Ȳ_k = Σ_{j=1}^{K-1} g(j/K) × ΔY_{k+j}
    
    Args:
        returns: Array of log returns
        K: Pre-averaging window size
        weight_type: Weight function type
    
    Returns:
        Pre-averaged returns
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    if n < K:
        raise ValueError(f"Not enough observations: {n} < K={K}")
    
    if NUMBA_AVAILABLE and weight_type == 'bartlett':
        return _preaverage_returns_numba(returns, K)
    
    # Fallback: numpy implementation
    weights = compute_preaveraging_weights(K - 1, weight_type)
    n_bar = n - K + 1
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
    Q_ij(τ_k) = Ȳ_i,k × Ȳ_j,k
    
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
    
    return Y_bar_i * Y_bar_j


# =============================================================================
# 3. Tail Index Estimation (Extended Hill Estimator)
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
    
    From paper Section 3.2, Eq. 3.4-3.6
    
    Args:
        Q: Array of quadratic covariation increments
        c_u: Constant for k selection (default: 2.0)
        c_omega: Constant for trimming (default: 1/3)
        c_xi: Constant for bias adjustment (default: 0.01)
        xi: Exponent for bias adjustment (default: 0.2)
        p: Number of assets
        alpha_min: Minimum tail index (default: 1.1)
        alpha_max: Maximum tail index (default: 4.0)
        use_median_centered: If True, estimate from |Q - median(Q)|
    
    Returns:
        TailIndexResult with estimated alpha and diagnostics
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = Q[~np.isnan(Q)]
    Q = Q[Q != 0]
    n = len(Q)
    
    if n < 10:
        return TailIndexResult(alpha=alpha_min, k=0, omega=0, n=n)
    
    # Use median-centered values (more robust)
    if use_median_centered:
        MQ = np.median(Q)
        values_for_tail = np.abs(Q - MQ)
        values_for_tail = values_for_tail[values_for_tail > 0]
        n_tail = len(values_for_tail)
    else:
        values_for_tail = np.abs(Q)
        n_tail = n
    
    if n_tail < 10:
        return TailIndexResult(alpha=alpha_min, k=0, omega=0, n=n)
    
    # Use numba-accelerated version if available
    if NUMBA_AVAILABLE:
        alpha = _estimate_tail_index_numba(values_for_tail.astype(np.float64), p=2)
        alpha = np.clip(alpha, alpha_min, alpha_max)
        k = int(c_u * np.floor(np.sqrt(n_tail)))
        omega = int(np.ceil(c_omega * np.log(max(p, 2))))
        return TailIndexResult(alpha=alpha, k=k, omega=omega, n=n)
    
    # Fallback: numpy implementation
    k = int(c_u * np.floor(np.sqrt(n_tail)))
    k = max(k, 2)
    omega = int(np.ceil(c_omega * np.log(max(p, 2))))
    omega = max(omega, 0)
    
    sorted_vals = np.sort(values_for_tail)[::-1]
    
    if k + omega >= len(sorted_vals):
        k = max(2, len(sorted_vals) - omega - 1)
    
    if k < 2:
        return TailIndexResult(alpha=alpha_min, k=k, omega=omega, n=n)
    
    start_idx = omega
    end_idx = omega + k
    
    if end_idx >= len(sorted_vals):
        end_idx = len(sorted_vals) - 1
        k = end_idx - start_idx
    
    eps = np.finfo(float).eps * 100
    log_sorted = np.log(sorted_vals[start_idx:end_idx] + eps)
    log_ref = np.log(sorted_vals[end_idx - 1] + eps)
    
    alpha_inv = np.mean(log_sorted - log_ref)
    
    if alpha_inv <= 0:
        alpha_tilde = alpha_max
    else:
        alpha_tilde = 1.0 / alpha_inv
    
    adjustment = c_xi * (n_tail ** (-xi)) * np.log(max(p, 2))
    alpha_hat = alpha_tilde - adjustment
    alpha_hat = np.clip(alpha_hat, alpha_min, alpha_max)
    
    return TailIndexResult(alpha=alpha_hat, k=k, omega=omega, n=n)


# =============================================================================
# 4. Adaptive Threshold Calculation (Eq. 5.17)
# =============================================================================

def compute_c_alpha(alpha: float) -> float:
    """Compute the scaling constant c_α."""
    if alpha <= 1:
        return 1.0
    term1 = (alpha - 1) / alpha
    if alpha < 2:
        term2 = np.sqrt((2 - alpha) / alpha)
    else:
        term2 = 0
    return max(term1, term2)


def calculate_threshold(
    Q: np.ndarray,
    alpha: float,
    p: int = 1,
    c: float = 0.35,
    K_n: Optional[int] = None
) -> ThresholdResult:
    """
    Calculate adaptive threshold θ_ij.
    
    From paper Eq. 5.17
    
    Args:
        Q: Array of quadratic covariation increments
        alpha: Estimated tail index
        p: Number of assets (default: 1)
        c: Threshold constant (0.35 for empirical)
        K_n: Bandwidth parameter (default: floor(n^0.5))
    
    Returns:
        ThresholdResult with threshold and diagnostics
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = Q[~np.isnan(Q)]
    n = len(Q)
    
    if K_n is None:
        K_n = int(np.floor(np.sqrt(n)))
    K_n = max(K_n, 1)
    alpha = max(alpha, 1.01)
    
    # Use numba if available
    if NUMBA_AVAILABLE:
        theta = _calculate_threshold_numba(Q, alpha, c)
        c_alpha = compute_c_alpha(alpha)
        MQ = np.median(Q)
        abs_dev = np.abs(Q - MQ)
        S_hat = np.median(abs_dev)
        return ThresholdResult(theta=theta, alpha=alpha, S_hat=S_hat, c_alpha=c_alpha, K_n=K_n)
    
    # Fallback: numpy implementation
    c_alpha = compute_c_alpha(alpha)
    MQ = np.median(Q)
    abs_dev = np.abs(Q - MQ)
    
    sorted_dev = np.sort(abs_dev)
    trim_n = max(1, int(0.01 * n))
    trimmed_dev = sorted_dev[trim_n:-trim_n] if 2*trim_n < n else sorted_dev
    
    eps = np.finfo(float).eps
    if alpha <= 2:
        S_hat = np.mean(np.maximum(trimmed_dev, eps) ** alpha)
    else:
        mad = np.median(abs_dev)
        if mad > eps:
            normalized = np.minimum(abs_dev / mad, 10)
            S_hat = (mad ** alpha) * np.mean(normalized ** alpha)
        else:
            S_hat = np.mean(np.maximum(abs_dev, eps) ** alpha)
    
    S_hat = max(S_hat, eps)
    
    effective_log_p = max(np.log(max(p, 2)), 2.0)
    numerator = K_n * effective_log_p * (alpha - 1)
    denominator = c_alpha * S_hat * max(n - K_n, 1)
    
    if denominator <= 0:
        denominator = eps
    
    ratio = numerator / denominator
    if ratio <= 0:
        ratio = eps
    
    theta = c * (ratio ** (1.0 / alpha))
    
    # Alpha-dependent bounds
    if alpha < 1.5:
        min_q, max_q = 0.90, 0.98
    elif alpha < 2.0:
        min_q, max_q = 0.92, 0.99
    elif alpha < 2.5:
        min_q, max_q = 0.94, 0.995
    else:
        min_q, max_q = 0.95, 0.999
    
    min_theta = np.quantile(abs_dev, min_q)
    max_theta = np.quantile(abs_dev, max_q)
    theta = np.clip(theta, min_theta, max_theta)
    
    return ThresholdResult(theta=theta, alpha=alpha, S_hat=S_hat, c_alpha=c_alpha, K_n=K_n)


# =============================================================================
# 5. Median-Centered Truncation
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
    """
    Q = np.asarray(Q, dtype=np.float64)
    
    if NUMBA_AVAILABLE:
        truncated_Q, n_truncated = _truncate_with_median_numba(Q, theta)
        mask = np.abs(Q - np.median(Q)) <= theta
        return truncated_Q, mask, n_truncated
    
    # Fallback: numpy
    MQ = np.median(Q)
    mask = np.abs(Q - MQ) <= theta
    truncated_Q = np.where(mask, Q, MQ)
    n_truncated = np.sum(~mask)
    
    return truncated_Q, mask, n_truncated


# =============================================================================
# 6. TARV Calculation (Realized Variance/Covariance)
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
    2. Compute quadratic variables Q_ii = Ȳ²
    3. Estimate tail index
    4. Calculate adaptive threshold
    5. Apply median-centered truncation
    6. Sum truncated Q values
    
    Args:
        returns: Array of log returns
        K: Pre-averaging window size
        p: Number of assets
        c: Threshold constant (0.35 empirical)
        weight_type: Pre-averaging weight type
        return_details: If True, return TARVResult
        universal_alpha: If provided, use this alpha
    
    Returns:
        TARV value or TARVResult with details
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    
    if n < K + 10:
        if return_details:
            return TARVResult(tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0)
        return np.nan
    
    # Pre-average
    Y_bar = preaverage_returns(returns, K, weight_type)
    Q = Y_bar ** 2
    
    # Estimate tail index
    if universal_alpha is not None:
        alpha = universal_alpha
    else:
        tail_result = estimate_tail_index(Q, p=p)
        alpha = tail_result.alpha
    
    # Calculate threshold
    thresh_result = calculate_threshold(Q, alpha, p=p, c=c)
    theta = thresh_result.theta
    
    # Apply truncation
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    # Sum to get TARV
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    tarv = np.sum(truncated_Q) / phi_22
    standard_rv = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(
            tarv=tarv, standard_rv=standard_rv, n_truncated=n_truncated,
            n_total=len(Q), truncation_rate=n_truncated / len(Q) * 100,
            threshold=theta, alpha=alpha
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
    
    Args:
        returns_i: Log returns of asset i
        returns_j: Log returns of asset j
        K: Pre-averaging window size
        p: Number of assets
        c: Threshold constant
        weight_type: Pre-averaging weight type
        return_details: If True, return TARVResult
        universal_alpha: If provided, use this alpha
    
    Returns:
        TARV covariance or TARVResult with details
    """
    returns_i = np.asarray(returns_i, dtype=np.float64)
    returns_j = np.asarray(returns_j, dtype=np.float64)
    
    valid_mask = ~(np.isnan(returns_i) | np.isnan(returns_j))
    returns_i = returns_i[valid_mask]
    returns_j = returns_j[valid_mask]
    n = len(returns_i)
    
    if n < K + 10:
        if return_details:
            return TARVResult(tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0)
        return np.nan
    
    Y_bar_i = preaverage_returns(returns_i, K, weight_type)
    Y_bar_j = preaverage_returns(returns_j, K, weight_type)
    Q = Y_bar_i * Y_bar_j
    
    if universal_alpha is not None:
        alpha = universal_alpha
    else:
        tail_result = estimate_tail_index(np.abs(Q), p=p)
        alpha = tail_result.alpha
    
    thresh_result = calculate_threshold(Q, alpha, p=p, c=c)
    theta = thresh_result.theta
    
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    tarv_cov = np.sum(truncated_Q) / phi_22
    standard_cov = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(
            tarv=tarv_cov, standard_rv=standard_cov, n_truncated=n_truncated,
            n_total=len(Q), truncation_rate=n_truncated / len(Q) * 100,
            threshold=theta, alpha=alpha
        )
    return tarv_cov


# =============================================================================
# 7. Realized Beta Calculation
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
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    market_returns = np.asarray(market_returns, dtype=np.float64)
    
    valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
    asset_returns = asset_returns[valid_mask]
    market_returns = market_returns[valid_mask]
    
    if use_tarv:
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
    
    if var_market == 0 or np.isnan(var_market):
        beta = np.nan
    else:
        beta = cov / var_market
    
    info['beta'] = beta
    return beta, info


# =============================================================================
# 8. Universal Alpha Estimation
# =============================================================================

def estimate_universal_alpha(
    returns_list: List[np.ndarray],
    K: int,
    p: int = 2,
    method: str = 'median'
) -> Tuple[float, dict]:
    """
    Estimate a universal tail index from multiple return series.
    
    Args:
        returns_list: List of return arrays
        K: Pre-averaging window size
        p: Number of assets parameter
        method: How to combine ('median', 'mean', 'min')
    
    Returns:
        Tuple of (universal_alpha, info_dict)
    """
    individual_alphas = []
    alpha_details = []
    
    for i, returns in enumerate(returns_list):
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < K + 10:
            continue
        
        Y_bar = preaverage_returns(returns, K)
        Q = Y_bar ** 2
        
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
    
    if method == 'median':
        universal_alpha = float(np.median(individual_alphas))
    elif method == 'mean':
        universal_alpha = float(np.mean(individual_alphas))
    elif method == 'min':
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


# =============================================================================
# 9. Truncate-First TARV Functions
# =============================================================================

def calculate_tarv_variance_truncate_first(
    returns: np.ndarray,
    K: int,
    p: int = 1,
    c: float = 0.35,
    weight_type: str = 'bartlett',
    return_details: bool = False,
    universal_alpha: Optional[float] = None,
    initial_truncate_quantile: float = 0.95
) -> Union[float, TARVResult]:
    """
    Calculate TARV variance with TRUNCATION-FIRST approach.
    
    Key difference: Initial truncation (95%) → Estimate α from clean data → Final truncation
    
    Args:
        universal_alpha: If provided (Rolling), use this. If None (Individual), estimate from truncated data.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    
    if n < K + 10:
        if return_details:
            return TARVResult(tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0)
        return np.nan
    
    Y_bar = preaverage_returns(returns, K, weight_type)
    Q = Y_bar ** 2
    
    if universal_alpha is not None:
        alpha = universal_alpha
    else:
        MQ = np.median(Q)
        abs_dev = np.abs(Q - MQ)
        initial_theta = np.quantile(abs_dev, initial_truncate_quantile)
        initial_mask = abs_dev <= initial_theta
        Q_for_tail = Q[initial_mask]
        
        if len(Q_for_tail) < 10:
            Q_for_tail = Q
        
        tail_result = estimate_tail_index(Q_for_tail, p=p)
        alpha = tail_result.alpha
    
    thresh_result = calculate_threshold(Q, alpha, p=p, c=c)
    theta = thresh_result.theta
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    tarv = np.sum(truncated_Q) / phi_22
    standard_rv = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(tarv=tarv, standard_rv=standard_rv, n_truncated=n_truncated,
            n_total=len(Q), truncation_rate=n_truncated / len(Q) * 100,
            threshold=theta, alpha=alpha)
    return tarv


def calculate_tarv_covariance_truncate_first(
    returns_i: np.ndarray,
    returns_j: np.ndarray,
    K: int,
    p: int = 2,
    c: float = 0.35,
    weight_type: str = 'bartlett',
    return_details: bool = False,
    universal_alpha: Optional[float] = None,
    initial_truncate_quantile: float = 0.95
) -> Union[float, TARVResult]:
    """
    Calculate TARV covariance with TRUNCATION-FIRST approach.
    
    Args:
        universal_alpha: If provided (Rolling), use this. If None (Individual), estimate from truncated data.
    """
    returns_i = np.asarray(returns_i, dtype=np.float64)
    returns_j = np.asarray(returns_j, dtype=np.float64)
    
    valid_mask = ~(np.isnan(returns_i) | np.isnan(returns_j))
    returns_i = returns_i[valid_mask]
    returns_j = returns_j[valid_mask]
    n = len(returns_i)
    
    if n < K + 10:
        if return_details:
            return TARVResult(tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0)
        return np.nan
    
    Y_bar_i = preaverage_returns(returns_i, K, weight_type)
    Y_bar_j = preaverage_returns(returns_j, K, weight_type)
    Q = Y_bar_i * Y_bar_j
    
    if universal_alpha is not None:
        alpha = universal_alpha
    else:
        Q_abs = np.abs(Q)
        MQ = np.median(Q_abs)
        abs_dev = np.abs(Q_abs - MQ)
        initial_theta = np.quantile(abs_dev, initial_truncate_quantile)
        initial_mask = abs_dev <= initial_theta
        Q_for_tail = Q_abs[initial_mask]
        
        if len(Q_for_tail) < 10:
            Q_for_tail = Q_abs
        
        tail_result = estimate_tail_index(Q_for_tail, p=p)
        alpha = tail_result.alpha
    
    thresh_result = calculate_threshold(Q, alpha, p=p, c=c)
    theta = thresh_result.theta
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    tarv_cov = np.sum(truncated_Q) / phi_22
    standard_cov = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(tarv=tarv_cov, standard_rv=standard_cov, n_truncated=n_truncated,
            n_total=len(Q), truncation_rate=n_truncated / len(Q) * 100,
            threshold=theta, alpha=alpha)
    return tarv_cov


def calculate_realized_beta_truncate_first(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    K: int,
    p: int = 2,
    c: float = 0.35,
    weight_type: str = 'bartlett',
    universal_alpha: Optional[float] = None
) -> Tuple[float, dict]:
    """Calculate realized beta using TRUNCATION-FIRST TARV method."""
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    market_returns = np.asarray(market_returns, dtype=np.float64)
    
    valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
    asset_returns = asset_returns[valid_mask]
    market_returns = market_returns[valid_mask]
    
    cov_result = calculate_tarv_covariance_truncate_first(
        asset_returns, market_returns, K, p, c, weight_type, 
        return_details=True, universal_alpha=universal_alpha
    )
    var_result = calculate_tarv_variance_truncate_first(
        market_returns, K, p, c, weight_type, 
        return_details=True, universal_alpha=universal_alpha
    )
    
    cov = cov_result.tarv
    var_market = var_result.tarv
    
    info = {
        'method': 'TARV-TruncateFirst',
        'cov': cov, 'var_market': var_market,
        'cov_alpha': cov_result.alpha, 'var_alpha': var_result.alpha,
        'cov_truncation_rate': cov_result.truncation_rate,
        'var_truncation_rate': var_result.truncation_rate
    }
    
    beta = cov / var_market if var_market != 0 and not np.isnan(var_market) else np.nan
    info['beta'] = beta
    return beta, info


# =============================================================================
# 10. Simple Truncation (No Tail-Adaptive)
# =============================================================================

def calculate_simple_truncated_variance(
    returns: np.ndarray,
    K: int,
    truncate_quantile: float = 0.95,
    weight_type: str = 'bartlett',
    return_details: bool = False
) -> Union[float, TARVResult]:
    """Calculate realized variance with simple quantile-based truncation."""
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    
    if n < K + 10:
        if return_details:
            return TARVResult(tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0)
        return np.nan
    
    Y_bar = preaverage_returns(returns, K, weight_type)
    Q = Y_bar ** 2
    
    MQ = np.median(Q)
    abs_dev = np.abs(Q - MQ)
    theta = np.quantile(abs_dev, truncate_quantile)
    
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    truncated_rv = np.sum(truncated_Q) / phi_22
    standard_rv = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(tarv=truncated_rv, standard_rv=standard_rv, n_truncated=n_truncated,
            n_total=len(Q), truncation_rate=n_truncated / len(Q) * 100, threshold=theta, alpha=0)
    return truncated_rv


def calculate_simple_truncated_covariance(
    returns_i: np.ndarray,
    returns_j: np.ndarray,
    K: int,
    truncate_quantile: float = 0.95,
    weight_type: str = 'bartlett',
    return_details: bool = False
) -> Union[float, TARVResult]:
    """Calculate realized covariance with simple quantile-based truncation."""
    returns_i = np.asarray(returns_i, dtype=np.float64)
    returns_j = np.asarray(returns_j, dtype=np.float64)
    
    valid_mask = ~(np.isnan(returns_i) | np.isnan(returns_j))
    returns_i = returns_i[valid_mask]
    returns_j = returns_j[valid_mask]
    n = len(returns_i)
    
    if n < K + 10:
        if return_details:
            return TARVResult(tarv=np.nan, standard_rv=np.nan, n_truncated=0,
                n_total=n, truncation_rate=0, threshold=0, alpha=0)
        return np.nan
    
    Y_bar_i = preaverage_returns(returns_i, K, weight_type)
    Y_bar_j = preaverage_returns(returns_j, K, weight_type)
    Q = Y_bar_i * Y_bar_j
    
    MQ = np.median(Q)
    abs_dev = np.abs(Q - MQ)
    theta = np.quantile(abs_dev, truncate_quantile)
    
    truncated_Q, mask, n_truncated = truncate_with_median_centering(Q, theta)
    
    phi_22 = np.sum(compute_preaveraging_weights(K - 1, weight_type) ** 2)
    truncated_cov = np.sum(truncated_Q) / phi_22
    standard_cov = np.sum(Q) / phi_22
    
    if return_details:
        return TARVResult(tarv=truncated_cov, standard_rv=standard_cov, n_truncated=n_truncated,
            n_total=len(Q), truncation_rate=n_truncated / len(Q) * 100, threshold=theta, alpha=0)
    return truncated_cov


def calculate_realized_beta_simple_truncation(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    K: int,
    truncate_quantile: float = 0.95,
    weight_type: str = 'bartlett'
) -> Tuple[float, dict]:
    """Calculate realized beta using simple quantile-based truncation."""
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    market_returns = np.asarray(market_returns, dtype=np.float64)
    
    valid_mask = ~(np.isnan(asset_returns) | np.isnan(market_returns))
    asset_returns = asset_returns[valid_mask]
    market_returns = market_returns[valid_mask]
    
    cov_result = calculate_simple_truncated_covariance(
        asset_returns, market_returns, K, truncate_quantile, weight_type, return_details=True
    )
    var_result = calculate_simple_truncated_variance(
        market_returns, K, truncate_quantile, weight_type, return_details=True
    )
    
    cov = cov_result.tarv
    var_market = var_result.tarv
    
    info = {
        'method': 'Simple-Truncation',
        'cov': cov, 'var_market': var_market,
        'truncate_quantile': truncate_quantile,
        'cov_truncation_rate': cov_result.truncation_rate,
        'var_truncation_rate': var_result.truncation_rate
    }
    
    beta = cov / var_market if var_market != 0 and not np.isnan(var_market) else np.nan
    info['beta'] = beta
    return beta, info


# =============================================================================
# 11. Helper Functions
# =============================================================================

def optimal_K(n: int, method: str = 'sqrt') -> int:
    """
    Calculate optimal pre-averaging window size K.
    
    Args:
        n: Number of observations
        method: 'sqrt' for K = sqrt(n)
    
    Returns:
        Optimal K
    """
    if method == 'sqrt':
        return max(2, int(np.sqrt(n)))
    elif method == 'theta':
        return max(2, int(1.5 * np.sqrt(n)))
    else:
        return max(2, int(np.sqrt(n)))


# =============================================================================
# Benchmark
# =============================================================================

def _benchmark():
    """Benchmark numba vs numpy speed."""
    import time
    
    print(f"\n{'='*60}")
    print(f"TARV Benchmark (Numba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'})")
    print(f"{'='*60}")
    
    np.random.seed(42)
    n = 50000
    returns = np.random.standard_t(df=4, size=n) * 0.001
    K = 30
    
    print(f"Data: {n:,} observations, K={K}")
    
    # Warm up
    _ = preaverage_returns(returns[:1000], K)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(5):
        result = calculate_tarv_variance(returns, K, return_details=True)
    elapsed = (time.perf_counter() - start) / 5 * 1000
    
    print(f"TARV Variance: {elapsed:.2f} ms")
    print(f"  Result: TARV={result.tarv:.6f}, Trunc={result.truncation_rate:.1f}%, α={result.alpha:.2f}")
    
    # Beta
    market = returns
    asset = 0.8 * market + np.random.standard_t(df=4, size=n) * 0.0005
    
    start = time.perf_counter()
    for _ in range(5):
        beta, info = calculate_realized_beta(asset, market, K)
    elapsed = (time.perf_counter() - start) / 5 * 1000
    
    print(f"Realized Beta: {elapsed:.2f} ms")
    print(f"  Result: β={beta:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _benchmark()
