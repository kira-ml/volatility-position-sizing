# Volatility Forecasting for Risk-Aware Position Sizing

**A machine learning approach to forward volatility estimation for dynamic position management.**

Investment returns depend as much on *how much* capital is deployed as on *which direction* is predicted. A correct directional view can still produce losses if a position is oversized during unexpectedly high volatility. Conversely, an undersized position during calm markets captures only a fraction of a high-conviction opportunity.

This project builds a supervised model to forecast near-term (5-day) realized volatility for individual equities. The forecast drives a dynamic position-sizing rule: `position_scale = target_vol / predicted_vol`. The objective is not to predict price direction, but to produce a statistically evaluated, well-calibrated volatility forecast that improves risk-adjusted portfolio performance relative to static position sizing.

---

## Why This Matters

Professional portfolio managers often operate under a target volatility mandate. Position sizing is the primary tool available to meet this mandate. An inaccurate volatility forecast harms the portfolio in two ways:

- **Underestimation:** Allocates too much capital. A volatility spike triggers a stop-out, causing a loss that exceeds the intended risk budget.
- **Overestimation:** Allocates too little capital. The market moves favorably, but the portfolio captures only a portion of the available return.

A robust, forward-looking volatility forecast supports consistent risk-adjusted performance, avoidance of unintended risk concentration, and efficient use of a limited risk budget.

---

## Project Objectives

1. Build a model that achieves lower out-of-sample forecast error (RMSE, MAE) against three progressively sophisticated baselines.
2. Verify that forecasts are statistically unbiased using the Mincer-Zarnowitz regression (testing the joint null hypothesis that α=0 and β=1).
3. Demonstrate that a dynamic position-sizing rule driven by the model achieves a higher Sharpe ratio and lower realized volatility deviation from a 15% annualized target versus static sizing in a historical simulation.
4. Visually assess forecast calibration using a volatility cone.

---

## Data

- **Equity data:** Daily OHLCV for 10–20 liquid US large-cap stocks (e.g., AAPL, MSFT, GOOGL, AMZN, META, JPM, XOM, JNJ, WMT, TSLA) sourced via `yfinance`.
- **Market context:** ^VIX index data for market-level volatility regime information sourced via `yfinance`.
- **Period:** Approximately 5–10 years of daily data.

---

## Methodology

### Target Definition

The prediction target is 5-day forward annualized realized volatility, calculated as the standard deviation of daily log returns over the subsequent 5 trading days, scaled to an annual figure.

### Feature Categories

- **Historical realized volatility:** Rolling windows at 5, 10, 21, and 63-day horizons
- **Range-based estimators:** Parkinson volatility
- **Volume-weighted metrics:** Average True Range (ATR) percentile rank
- **Market context:** VIX closing level
- **Cross-sectional:** Sector-average EWMA volatility

### Model Evaluation Framework

Evaluation follows strict temporal ordering to avoid look-ahead bias. A purged walk-forward cross-validation scheme is used to prevent information leakage between adjacent training and testing windows.

- **Error metrics:** RMSE and MAE against the baseline ladder
- **Unbiasedness:** Mincer-Zarnowitz regression (joint F-test for α=0, β=1)
- **Tail error:** 95th percentile of absolute forecast error
- **Calibration:** Volatility cone visualization

---

## Baseline Solutions

A ladder of three progressively stronger baselines is established before considering any advanced model. Each baseline adds a logical refinement. Complexity is justified only when it demonstrably outperforms simpler alternatives on pre-specified criteria.

| Baseline | Description | Rationale |
|---|---|---|
| **1. Rolling Historical Volatility** | 21-day annualized standard deviation of log returns. | Simplest defensible heuristic. Represents the null hypothesis that recent history contains all relevant information. |
| **2. EWMA Volatility** | Exponentially weighted moving average with λ=0.94 (RiskMetrics standard). | Captures volatility clustering by weighting recent observations more heavily. The minimal model that encodes this empirical regularity. |
| **3. Ridge Regression** | L2-penalized linear model trained on curated features. | Tests whether a linear combination of well-chosen predictors is sufficient. If this baseline is competitive, it is preferred on grounds of interpretability and simplicity. |

---

## Model

### Primary Approach (Conditional on Baseline Results)

If the Ridge regression baseline does not adequately capture the relationship between features and future volatility, a simple averaged ensemble of **Random Forest** and **LightGBM** is evaluated. Random Forest uses bagging to reduce variance. LightGBM uses boosting to reduce bias. Averaging the two can produce forecasts that are more stable out-of-sample than either model alone.

### Adoption Criteria

The ensemble is only adopted if both of the following conditions are met:
1. A statistical test for non-linearity (Ramsey's RESET test applied to Ridge residuals) rejects the null of correct linear specification.
2. The Ridge regression baseline does not achieve a Mincer-Zarnowitz β of at least 0.90.

If both conditions are not met, the Ridge regression baseline is the final model. This finding is itself informative: it demonstrates that a well-specified linear model is sufficient for the problem.

### Evaluation of the Ensemble (if adopted)

The ensemble is compared to the Ridge baseline on the same out-of-sample periods. It must demonstrate:
- A Mincer-Zarnowitz β meaningfully closer to 1
- A reduction in tail errors of at least 5%
- A statistically significant improvement in RMSE via the Diebold-Mariano test
- Either a higher Sharpe ratio (difference > 0.1) or a meaningfully lower maximum drawdown (>10% relative reduction)

---

## Backtest: Dynamic vs. Static Position Sizing

A historical simulation compares two strategies on the same out-of-sample period:

1. **Static sizing:** A constant dollar amount is allocated to each position daily.
2. **Dynamic sizing:** Position size is scaled as `position_scale = target_vol / predicted_vol`, capped at a maximum leverage threshold.

The simulation assumes a target volatility of 15% annualized. Evaluation metrics are Sharpe ratio, realized volatility, maximum drawdown, and absolute deviation of realized volatility from the 15% target.

---

## Repository Structure

```
├── data/                   # Raw and processed data (gitignored)
├── notebooks/              # Jupyter notebooks for exploration and visualization
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_model.ipynb
│   └── 05_backtest_analysis.ipynb
├── src/                    # Modular source code
│   ├── features.py         # Feature engineering pipeline
│   ├── baselines.py        # Baseline model implementations
│   ├── train.py            # Model training and cross-validation
│   ├── evaluate.py         # Evaluation metrics and statistical tests
│   └── backtest.py         # Position sizing simulation
├── README.md               # Project overview and findings
└── requirements.txt        # Python dependencies
```

---

## Key Findings

*To be completed after analysis.*

*This section will summarize:*
- *Performance of each baseline relative to the others*
- *Whether the Ridge regression baseline was sufficient or the ensemble was adopted*
- *Mincer-Zarnowitz regression results for the final model (was the forecast unbiased?)*
- *Sharpe ratio comparison between dynamic and static sizing*
- *Any regime-dependent behavior observed in the volatility cone*
- *Documented failure modes*

---

## Limitations & Assumptions

The following limitations are explicitly acknowledged:

- **Single-horizon forecasting:** Only 5-day forward volatility is modeled. The volatility term structure across horizons is not addressed.
- **Univariate scope:** Each stock is modeled independently. Cross-asset dependencies are captured only through market-level and sector-level features. Full covariance modeling is a separate undertaking.
- **No transaction costs:** The backtest simulation does not include trading costs or market impact. It is a measurement tool for forecast quality, not a profitability claim.
- **No explicit regime-switching:** The model does not explicitly model discrete volatility regimes. It relies on features like VIX for continuous regime context.
- **Stationarity assumption:** Feature relationships are assumed stable enough for walk-forward validation to capture. Structural breaks represent model risk that is acknowledged but not modeled.
- **Liquidity assumption:** Selected large-cap stocks are assumed sufficiently liquid that position sizes do not materially impact market prices.

---

## References

- Andersen, T. G., & Bollerslev, T. (1998). Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts. *International Economic Review*.
- Mincer, J. A., & Zarnowitz, V. (1969). The Evaluation of Economic Forecasts. In *Economic Forecasts and Expectations*.
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. *Journal of Business*.
- Poon, S.-H., & Granger, C. W. J. (2003). Forecasting Volatility in Financial Markets: A Review. *Journal of Economic Literature*.
- RiskMetrics Group. (1996). *RiskMetrics — Technical Document*.

---

## Skills Demonstrated

- Volatility estimation and forecasting for second-moment prediction
- Time-series feature engineering grounded in the volatility literature
- Model evaluation beyond error metrics: unbiasedness testing, calibration assessment, and tail error analysis
- Baseline-first methodology with explicit adoption criteria for complexity
- Risk-based position sizing with economic utility measurement
- Purged walk-forward cross-validation for time-series data
