# Measuring the Low-Beta Anomaly: High-Frequency Evidence from Equities and Cryptocurrencies

**Team 4 Research Project**  
Woohyuk Choi, Seonghyun Yang

---

## 📋 Project Overview

This research investigates the **low-beta anomaly** using high-frequency data from both equity and cryptocurrency markets. We employ advanced realized volatility estimation techniques to distinguish genuine market anomalies from measurement artifacts.

### Research Motivation

The traditional CAPM predicts a linear relationship between beta and expected returns. However, empirical evidence shows that low-beta assets often outperform high-beta assets on a risk-adjusted basis - this is known as the **low-beta anomaly**.

While this anomaly is well-documented in equity markets (Frazzini & Pedersen, 2014), its existence in cryptocurrency markets remains unexplored. Previous studies using daily or weekly data found no significant beta effects in crypto markets, but these null results may reflect measurement error rather than true absence of the anomaly.

---

## 🔬 Research Questions

**H1:** High-frequency beta estimates will significantly differ from daily estimates
- Expected to be larger in volatile crypto markets than equities

**H2:** Optimal estimation frequency exists where measurement precision is maximized
- Trade-off between more observations vs. microstructure noise
- May differ between equities and cryptocurrencies

**H3:** Accurate beta measurement clarifies whether the low-beta anomaly is real or a measurement artifact
- Does the anomaly exist in crypto when beta is properly measured?

---

## 🎯 Expected Contributions

1. **Methodological Framework**: Distinguish measurement artifacts from genuine anomalies
2. **First Rigorous Test**: Low-beta anomaly in cryptocurrency markets using high-frequency data
3. **Quantification**: Measure anomaly returns attributable to beta mismeasurement
4. **Cross-Market Analysis**: Compare structural differences between equities and cryptocurrencies

---

## 🛠️ Research Framework

### Step 1: Tail Index Estimation
- Estimate asset-specific tail heaviness using extended Hill estimator
- Addresses heterogeneous heavy tails across assets
- Determines appropriate truncation levels

### Step 2: Tail-Adaptive Realized Covariance (TARV)
- Apply adaptive truncation based on tail indices
- Filter microstructure noise, jumps, and heavy-tailed spikes
- Compute realized covariance and variance

### Step 3: Realized Volatility Aggregation
- Daily and multi-horizon volatility estimation
- Compare TARV vs. standard RV

### Step 4: Volatility-Sorted Portfolios
- Sort assets into quintiles/deciles by past-month volatility
- Construct Low-High spread portfolios
- Compare performance under TARV vs. standard RV

### Step 5: Multi-Scale Analysis
- Test anomaly across different time horizons (1-min to monthly)
- Identify optimal estimation frequency

### Step 6: Cross-Market Comparison
- Equities: NYSE/Nasdaq
- Cryptocurrencies: Binance/Upbit
- Structural difference analysis

### Step 7: Price Discovery Analysis
- Hasbrouck Information Shares
- Gonzalo-Granger Component Shares
- Lead-lag relationships

---

## 📊 Data Sources

### Equities (NYSE/Nasdaq)
- **Time-series**: Daily, hourly, minute bars
- **High-frequency**: 
  - L1: Best bid/offer (BBO)
  - L2: Full order book depth
- **Benchmark**: S&P 500

### Cryptocurrencies
- **Binance**
  - Time-series: Daily, hourly, minute bars
  - High-frequency: L1, L2, L3 (order-level)
  - Benchmark: CMC Top 10 Equal-Weighted Index
  
- **Upbit**
  - Time-series: Daily, hourly, minute bars
  - High-frequency: L1, L2, L3 (order-level)
  - Benchmark: Upbit Composite Index

### Currency Standardization
- Using USDT pairs for both Binance and Upbit
- Eliminates FX conversion noise

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- Sufficient disk space for high-frequency data storage

### Setup

Choose one of the following methods:

#### 1. Using pip (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Using Poetry
```bash
# Install dependencies
poetry install

# Activate environment
poetry shell
```

#### 3. Using Conda
```bash
# Create environment and install packages
conda env create -f environment.yml

# Activate environment
conda activate crypto-collector
```

---

## 📦 Required Packages

- `requests` - API communication for data collection
- `polars` - High-performance dataframe operations
- `pyarrow` - Arrow file format support
- `rich` - Enhanced terminal UI
- `numpy` - Numerical computations
- `scipy` - Statistical analysis
- `pandas` - Data manipulation (if needed)

---

## 🚀 Usage

### Data Collection

Run the collector script to gather high-frequency data:

```bash
python collector.py
```

The script will prompt you to:
1. Select exchange (Binance/Upbit/Both)
2. Choose data format (Arrow/Parquet/CSV)
3. Specify collection period
4. Enter trading pairs/market codes

### Analysis Pipeline

```bash
# Step 1: Estimate tail indices
python analysis/01_tail_estimation.py

# Step 2: Compute TARV
python analysis/02_tarv_computation.py

# Step 3: Construct portfolios
python analysis/03_portfolio_construction.py

# Step 4: Run anomaly tests
python analysis/04_anomaly_tests.py

# Step 5: Cross-market comparison
python analysis/05_cross_market_analysis.py
```

---

## 📁 Project Structure

```
.
├── collector.py              # Data collection script
├── requirements.txt          # Pip dependencies
├── pyproject.toml           # Poetry configuration
├── environment.yml          # Conda environment
├── README.md                # This file
├── data/                    # Collected market data
│   ├── binance/
│   ├── upbit/
│   └── equities/
├── analysis/                # Analysis scripts
│   ├── 01_tail_estimation.py
│   ├── 02_tarv_computation.py
│   ├── 03_portfolio_construction.py
│   ├── 04_anomaly_tests.py
│   └── 05_cross_market_analysis.py
├── results/                 # Analysis outputs
│   ├── figures/
│   ├── tables/
│   └── portfolios/
└── docs/                    # Documentation
    ├── methodology.md
    └── data_dictionary.md
```

---

## 📈 Expected Results

1. **High-frequency beta estimates** substantially reduce measurement error compared to daily estimates
2. **Optimal sampling frequency** differs between equities and cryptocurrencies due to microstructure differences
3. **Low-beta anomaly** may weaken in equities but emerge in cryptocurrencies under proper measurement
4. **Structural insights** into market efficiency and price discovery dynamics

---

## 📚 References

1. Black, F., Jensen, M. C., & Scholes, M. (1972). The capital asset pricing model: Some empirical tests.
2. Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.
3. Hollstein, F., Prokopczuk, M., & Wese Simen, C. (2020). The conditional capital asset pricing model revisited: Evidence from high-frequency betas. *Management Science*, 66(6), 2474-2494.
4. Liu, Y., Tsyvinski, A., & Wu, X. (2022). Common risk factors in cryptocurrency. *The Journal of Finance*, 77(2), 1133-1177.
5. Shin, M., Kim, D., & Fan, J. (2023). Adaptive robust large volatility matrix estimation based on high-frequency financial data. *Journal of Econometrics*, 237(1), 105514.

---

## ⚠️ Known Issues & Challenges

### Trading Hours Mismatch
- **Equities**: Limited trading hours (9:30 AM - 4:00 PM ET)
- **Crypto**: 24/7 continuous trading

**Potential Solutions**:
1. Restrict crypto data to equity market hours
2. Analyze overlapping vs. non-overlapping periods separately
3. Use rolling windows capturing both regimes

---

## 🤝 Contributing

This is a research project. For questions or collaboration inquiries, please contact:
- Woohyuk Choi
- Seonghyun Yang
