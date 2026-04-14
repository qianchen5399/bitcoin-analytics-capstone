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
## Data Sources

This project uses two primary data sources.

### Coin Metrics
Coin Metrics provides daily Bitcoin market and on-chain data used for:
- price context
- market structure
- accumulation regime analysis
- feature construction

### Polymarket
Polymarket provides prediction-market data used to construct:
- daily activity features
- liquidity-filtered “smart money” aggregates
- market participation and sentiment proxies

A central part of this project was testing whether these external market signals contain useful information for Bitcoin accumulation after proper filtering, aggregation, and leak-aware alignment.

---

## Method Overview

The project followed four broad stages:

1. **Data validation and temporal integrity checks**  
   Verified date coverage, repaired timestamp interpretation issues, and aligned Coin Metrics and Polymarket data at the daily level.

2. **Feature engineering**  
   Constructed Bitcoin market features and liquidity-filtered Polymarket activity features, with emphasis on daily aggregation and leak-aware design.

3. **Model development**  
   Compared multiple dynamic DCA strategy variants, including naive signal-driven versions, conservative overlays, and regime-conditional approaches.

4. **Backtesting and interpretation**  
   Evaluated the resulting strategies against uniform DCA using rolling one-year windows within the sponsor-provided evaluation framework.

---

## Final Model

The final model is a conservative, regime-conditional dynamic accumulation framework that uses liquidity-filtered Polymarket activity features as a conditional overlay within a long-only DCA structure.

Its purpose is not aggressive market timing. Instead, it seeks to modestly improve accumulation efficiency relative to uniform DCA while preserving interpretability, stability, and disciplined capital deployment.

---

## Limitations

Several limitations are important when interpreting these results:

- Polymarket history is much shorter than Bitcoin history
- raw external sentiment features are noisy and unstable
- the observed edge is modest rather than transformational
- backtest performance does not guarantee future outperformance
- results depend on historical relationships that may weaken across regimes

---

## Public Artifact

Public-facing project summary:  
**[Add public article / blog / LinkedIn post link here]**

This repository provides the full analytical support for the public-facing summary.

---

## Author

**Nick Chen**  
Georgia Institute of Technology  
Online Master of Science in Analytics (OMSA)

## Disclaimer

This repository was created for academic, research, and portfolio purposes as part of the Georgia Tech OMSA Practicum. It should not be interpreted as financial advice or a recommendation to buy or sell bitcoin.
