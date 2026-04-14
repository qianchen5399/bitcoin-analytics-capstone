# Bitcoin Analytics Practicum: Dynamic Bitcoin Accumulation with Prediction-Market Signals

This repository contains my Georgia Tech OMSA Practicum project on dynamic Bitcoin accumulation. The project studies whether external market information, especially liquidity-filtered prediction-market activity from Polymarket, can improve Bitcoin accumulation efficiency relative to a uniform Dollar-Cost Averaging (DCA) benchmark.

For final review, please start with:

- **`deliverables/bitcoin_practicum_final.ipynb`** — main final notebook

Note:
- **`deliverables/bitcoin_eda_modeling_support.ipynb`** is retained as a duplicate / backup copy of the main notebook.
- Additional exploratory and development materials are included elsewhere in the repository for context and reproducibility.

---

## Project Question

Can external signals improve Bitcoin accumulation without turning a disciplined DCA process into unstable market timing?

More specifically, this project evaluates whether Polymarket-derived features, when filtered, aggregated, and used conservatively, can provide useful conditioning signals for dynamic Bitcoin accumulation.


---


## Key Takeaway

Uniform DCA remained a strong and difficult-to-beat baseline throughout the project.

Raw prediction-market data did not appear useful as a direct timing signal on its own. However, after liquidity filtering, aggregation, and conservative regime-aware integration, Polymarket-derived activity features provided a modest but defensible improvement within a dynamic Bitcoin accumulation framework.

The main contribution of this project is not extreme outperformance, but a careful demonstration that external signals may add value only when they are structured thoughtfully, evaluated conservatively, and used as conditional overlays rather than aggressive predictors.

---

## Final Deliverables

The main final deliverables for this project are located in the `deliverables/` folder.

- **`deliverables/bitcoin_practicum_final.ipynb`**  
  Main final notebook containing the final research narrative, feature construction, modeling workflow, backtesting results, interpretation, and limitations.

- **`deliverables/EDA_Executive.ipynb`**  
  Condensed exploratory analysis highlighting the most important data validation and early findings.

- **`deliverables/EDA.ipynb`**  
  Full exploratory data analysis notebook with supporting visualizations and intermediate analysis.

- **`deliverables/regime_conditional_dca_v2.py`**  
  Final strategy implementation used in the main analysis.

- **`deliverables/backtest_template.py`**  
  Backtesting script used to evaluate model performance against the benchmark framework.

---

## What to Read First

If you are reviewing this repository for the first time, the recommended order is:

1. **`deliverables/bitcoin_practicum_final.ipynb`**  
   Main final analysis and conclusions

2. **`deliverables/EDA_Executive.ipynb`**  
   Shorter summary of the exploratory groundwork

3. **`deliverables/EDA.ipynb`**  
   Full exploratory analysis and supporting visuals

Other folders in the repository document earlier development history, templates, intermediate experiments, and reproducibility materials.

---



## Repository Overview

This repository provides a template and framework for:
1.  **Exploratory Data Analysis (EDA)** of Bitcoin price action and on-chain properties.
2.  **Feature Engineering** that integrates prediction market sentiment (Polymarket), macro indicators, and on-chain metrics.
3.  **Strategy Development** for daily purchase schedules (dynamic DCA).
4.  **Backtesting & Evaluation** against uniform DCA benchmarks.

### Repository Structure

```text

├── deliverables/                    # FINAL PROJECT MATERIALS (Start here)
│   ├── bitcoin_practicum_final.ipynb # Main final notebook for review
│   ├── bitcoin_eda_modeling_support.ipynb # Duplicate / backup of the main notebook
│   ├── EDA.ipynb                    # Full exploratory data analysis
│   ├── EDA_Executive.ipynb          # Condensed executive-style EDA
│   ├── regime_conditional_dca_v2.py # Final strategy implementation
│   └── backtest_template.py         # Backtesting script used for evaluation
├── data/                            # Bitcoin and Polymarket source data
│   ├── Coin Metrics/                # BTC market and on-chain data
│   ├── Polymarket/                  # Prediction-market data
│   └── download_data.py             # Data download utility
├── eda/                             # EDA DEVELOPMENT HISTORY
│   ├── EDA.ipynb                    # Main EDA working notebook
│   ├── EDA_Executive.ipynb          # Executive EDA working version
│   ├── EDA_v6.ipynb                 # Intermediate EDA version
│   ├── audit_raw_timestamps.py      # Timestamp auditing and validation
│   ├── bitcoin_eda_modeling_support.ipynb # Intermediate support notebook
│   ├── btc_polymarket_eda_pipeline.py # EDA pipeline script
│   ├── eda_starter_template.md      # Initial EDA guidance
│   ├── eda_starter_template.py      # Starter EDA template
│   ├── gt_msa_s26_eda_outline.md    # EDA outline and project framing
│   └── initial_EDA_*.ipynb          # Earlier exploratory notebook versions
├── template/                        # MODEL DEVELOPMENT HISTORY
│   ├── *.py                         # Iterative model strategy files and templates
│   └── *.md                         # Documentation for model logic and experiments
├── example_1/                       # REFERENCE EXAMPLE MATERIALS
│   └── ...                          # Example strategy and usage files
├── tests/                           # Tests and validation utilities
├── requirements.txt                 # Project dependencies
├── LICENSE                          # Repository license
└── README.md                        # Project overview and navigation
```
---

## Getting Started

### 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template
    cd bitcoin-analytics-capstone-template
    ```

2.  **Setup environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

### 2. Data Acquisition

The `data/` directory contains historical BTC price data and specific Polymarket datasets (Politics, Finance, Crypto).

Data can be [downloaded manually from Google Drive](https://drive.google.com/drive/folders/1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7?usp=sharing) into the `data/` folder, or you can use the automated script:

```bash
python data/download_data.py
```

**Included Data:**
* **CoinMetrics BTC Data**: Daily OHLCV and network metrics.
  * **Bitcoin Price Source of Truth**: The `PriceUSD` column in the CoinMetrics data is the source of truth for BTC-USD prices. This is renamed to `PriceUSD_coinmetrics` in the codebase. This is the only column you hypothetically need to build a model (along with the datetime index, of course).
* **Polymarket Data**: High-fidelity parquet files containing trades, odds history, and market metadata.
  * **Timestamp note**: Some parquet timestamp columns are stored with incorrect
    units (millisecond values encoded as microseconds). Direct reads can show
    dates near 1970. Use the built-in loaders in `template/prelude_template.py`
    or `eda/eda_starter_template.py`, which detect and correct these values at
    runtime.

**External Data:**
External data is encouraged; students are responsible for ensuring that the data license permits all project participants to access and use (i.e., no proprietary data).

**System Requirements:**
Assume a modern laptop specification (think 16GB M4 Air).

---

## Model Development Guidelines

The framework includes a **Template Baseline** in `template/`. This serves as a starting point, currently implementing a simple 200-day Moving Average filter (accumulating more when price is below the MA).

### Exploration Path: Prediction Market Integration

A core opportunity lies in evolving this baseline into a market-aware strategy, perhaps by leveraging **Polymarket data**.

**Illustrative Examples:**
*   **Election Probabilities**: You might investigate if political event probabilities correlate with BTC volatility.
*   **Economic Indicators**: Consider checking if prediction markets for Fed rate cuts act as leading indicators.
*   **Retail Sentiment**: Specific "Polymarket Crypto" markets could potentially serve as proxies for retail sentiment or exuberance.

### Running Backtests

**Backtest Date Range:**
* **Range:** `2018-01-01` to `2025-12-31` (inclusive; daily frequency; no days should be missing)
* The backtest engine uses rolling 1-year windows starting from the start date, generating daily windows until the end date.

**Baseline Model:**
```bash
python -m template.backtest_template
```

**Reference Implementation (Example 1):**
```bash
python -m example_1.run_backtest
```

---

## Key Performance Indicators

When evaluating strategies, you might consider the following metrics (which are calculated by the automated backtest engine):

1.  **Win Rate**: Useful for understanding consistency—how often does the strategy outperform a standard DCA over 1-year windows?
2.  **SPD (Sats Per Dollar)**: A measure of raw efficiency—are you acquiring more bitcoin for the same capital?
3.  **Model Score**: A composite metric that balances performance (Win Rate) with risk-adjusted returns, offering a holistic view of strategy health.

## Licensing

*   **Code:** This repository, including its analysis and documentation, is open-sourced under the **MIT License**.
*   **Data:** The data provided (e.g., CoinMetrics, Polymarket) is not covered by the MIT license and retains its original licensing terms. Please refer to the respective data providers for their terms of use.

---

## Contacts & Community

* **App:** [stackingsats.org](https://www.stackingsats.org/)
* **Website:** [trilemma.foundation](https://www.trilemma.foundation/)
* **Foundation:** [Trilemma Foundation](https://github.com/TrilemmaFoundation)
