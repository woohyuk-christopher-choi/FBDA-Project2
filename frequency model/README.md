# Frequency Model: Low-Beta Anomaly Analysis

A module for Low-Beta Anomaly analysis based on TARV (Tail-Adaptive Realized Volatility).

## File Structure

```
frequency model/
├── run_crypto_analysis.py   # Cryptocurrency analysis executable
├── run_stock_analysis.py    # Stock analysis executable
├── data_loader.py           # Cryptocurrency data loader (Binance)
├── stock_data_loader.py     # Stock data loader (S&P500)
├── tarv.py                  # TARV core functions
├── modeling.py              # Modeling utilities
├── portfolio.py             # Portfolio construction
└── README.md                # This document
```

## Quick Start

### Run Cryptocurrency Analysis
```bash
cd "frequency model"
python run_crypto_analysis.py
```

### Run Stock Analysis
```bash
cd "frequency model"
python run_stock_analysis.py
```

## Methodology

### Comparison of 6 Beta Estimation Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **TARV Individual alpha** | Original TARV (tail-first), per-asset alpha estimation |
| 2 | **TARV Rolling alpha** | Original TARV with rolling universal alpha across assets |
| 3 | **Truncate-First Ind alpha** | Truncation before tail estimation, per-asset alpha |
| 4 | **Truncate-First Roll alpha** | Truncation before tail estimation, rolling alpha |
| 5 | **Simple Truncation** | Fixed 95% quantile truncation (non-adaptive) |
| 6 | **Standard RV** | No truncation, simple realized variance/covariance |

### What is TARV?
- **Tail-Adaptive Realized Volatility** (Shin, Kim & Fan, 2023)
- Volatility estimation considering heavy-tail characteristics of high-frequency financial data
- Estimates tail index (alpha) using Hill estimator
- Performs adaptive truncation based on alpha value

## Analysis Parameters

### Cryptocurrency (Crypto) - 24/7 Trading
| Frequency | Window | Obs/Day | Window Obs | K (approx) |
|-----------|--------|---------|------------|------------|
| 5m | 30 days | 288 | 8,640 | 30 |
| 15m | 30 days | 96 | 2,880 | 17 |
| 30m | 30 days | 48 | 1,440 | 12 |
| 1h | 30 days | 24 | 720 | 8 |
| 1d | 120 days | 1 | 120 | 3 |

### Stock - US Market (9:30-16:00, 390 min/day)
| Frequency | Window | Obs/Day | Window Obs | K (approx) |
|-----------|--------|---------|------------|------------|
| 5min | 30 days | 78 | 2,340 | 16 |
| 15min | 30 days | 26 | 780 | 9 |
| 30min | 30 days | 13 | 390 | 6 |
| 1h | 30 days | 6 | 180 | 4 |
| 1d | 120 days | 1 | 120 | 3 |

### Common Settings
- **Rebalance**: Weekly (every Monday)
- **Holding Period**: 7 days
- **Portfolio**: 5 quintiles (Low_Beta, Q2, Q3, Q4, High_Beta)
- **BAB Strategy**: Long Low_Beta, Short High_Beta

## Output Files

### Cryptocurrency Analysis (`run_crypto_analysis.py`)
- `crypto_analysis_results.csv` - Detailed results
- `crypto_analysis_summary.csv` - Summary statistics
- `crypto_bab_returns.csv` - BAB returns by method and frequency

### Stock Analysis (`run_stock_analysis.py`)
- `stock_analysis_results.csv` - Detailed results
- `stock_analysis_summary.csv` - Summary statistics
- `stock_bab_returns.csv` - BAB returns by method and frequency

## Core Functions (tarv.py)

```python
# Original TARV-based beta calculation
calculate_realized_beta(asset_returns, market_returns, K, use_tarv=True, universal_alpha=None)

# Truncate-First beta calculation
calculate_realized_beta_truncate_first(asset_returns, market_returns, K, universal_alpha=None)

# Simple Truncation beta calculation
calculate_realized_beta_simple_truncation(asset_returns, market_returns, K)

# Universal alpha estimation (for rolling methods)
estimate_universal_alpha(returns_list, K, method='median')
```

## Low-Beta Anomaly

**Low-Beta Anomaly** refers to the phenomenon where assets with lower beta achieve higher 
risk-adjusted returns than high-beta assets, contrary to CAPM theory.

### BAB (Betting Against Beta) Strategy
- **Long**: Low-Beta portfolio
- **Short**: High-Beta portfolio
- **Positive BAB return** = Evidence of Low-Beta Anomaly

## References

- Shin, M., Kim, D., & Fan, J. (2023). *Tail-Adaptive Realized Volatility*
- Frazzini, A., & Pedersen, L. H. (2014). *Betting Against Beta*. Journal of Financial Economics

## Dependencies

```
numpy
pandas
scipy
```

## Contact

FBDA Project Team - 2025
