# Problem Framing: Volatility Forecasting for Risk-Aware Position Sizing

## 1. The Real-World Problem

Investment returns depend not only on predicting price direction but also on how much capital is allocated to each position.

A systematic portfolio manager with a correct directional view can still lose money if a position is oversized during a period of unexpectedly high volatility. The stock triggers a stop-loss, the position is closed, and the portfolio realizes a loss exceeding its intended risk budget.

The reverse also causes harm. A conservative position during an unexpectedly calm market wastes a high-conviction view. The portfolio captures only a fraction of the available return, creating a persistent drag relative to its benchmark.

Many systematic strategies use static position sizing or reactive rules that adjust slowly to changing conditions. This results in uneven risk contribution over time and inefficient use of a limited risk budget.

The core problem is not directional forecasting. It is **volatility forecasting for risk management**.

---

## 2. Why This Problem Matters

Professional portfolio managers often operate under a target volatility mandate. An institutional investor may require that the portfolio does not exceed 10% or 15% annualized volatility. Position sizing is the primary tool available to meet this mandate.

An inaccurate volatility forecast damages the portfolio in two measurable ways:

- **Underestimation of volatility:** The model predicts calm conditions and allocates a large position. Realized volatility rises. The position hits a stop-loss. The single-position loss violates the portfolio-level risk budget and contributes disproportionately to drawdowns.
- **Overestimation of volatility:** The model predicts turbulent conditions and allocates a small position. Realized volatility remains low. The market moves favorably. The portfolio captures an inadequate share of the return and underperforms its benchmark.

Neither outcome requires the manager to be wrong about price direction. Both are a function of poor volatility assessment.

A robust, forward-looking volatility forecast supports:
- Consistent risk-adjusted performance
- Avoidance of unintended risk concentration
- Efficient use of a limited risk budget
- Demonstrable risk awareness

For a data scientist entering quantitative finance, this problem represents a shift from predicting direction (a Level 1 problem) to quantifying uncertainty about that direction (a Level 2 problem). That shift indicates familiarity with how professional risk management functions in practice.

---

## 3. Project Objectives

This project has four measurable objectives, ordered from statistical to practical:

1. **Forecast Accuracy:** Build a machine learning model that achieves lower out-of-sample prediction error (RMSE, MAE) compared to three progressively sophisticated baselines: a 21-day rolling historical volatility, an exponentially weighted moving average (EWMA) with λ=0.94, and a ridge-regularized linear model with curated features.

2. **Statistical Unbiasedness:** Verify that the model's forecasts are unbiased using the Mincer-Zarnowitz regression. The joint null hypothesis that α=0 and β=1 should not be rejected at standard significance levels. An unbiased forecast means high predictions are not systematically too high and low predictions are not systematically too low.

3. **Economic Utility:** Demonstrate that a dynamic position-sizing rule driven by the model forecast achieves a higher Sharpe ratio and lower realized volatility deviation from a 15% annualized target than a static sizing rule over the same out-of-sample period. This tests whether statistical improvement translates into a meaningful practical outcome.

4. **Calibration Assessment:** Confirm that forecast errors fall within expected dispersion using a volatility cone visualization. A well-calibrated forecast should show realized volatility converging toward the prediction as the horizon extends.

---

## 4. Scope

### In-Scope

- 10–20 highly liquid US large-cap equities, selected to ensure data quality and avoid market microstructure noise
- A single prediction target: 5-day forward annualized realized volatility
- A focused feature set: lagged realized volatilities, range-based estimators (Parkinson), volume-weighted metrics, VIX context, and sector-average volatility
- A rigorous temporal evaluation framework using purged walk-forward cross-validation
- A ladder of three baselines (rolling historical, EWMA, ridge regression) to justify any added complexity
- A simple averaging ensemble of Random Forest and LightGBM as the advanced approach, adopted only if specific trigger conditions are met
- A clear connection from model forecast to a concrete position-sizing rule

### Intentionally Out-of-Scope

- Portfolio-level optimization or covariance matrix forecasting. This introduces dimensionality and estimation complexity orthogonal to the core volatility forecasting signal being evaluated.
- High-frequency or intraday data. The 5-day horizon is well-served by daily data; microstructure noise adds no signal at this frequency.
- Deep learning architectures (LSTMs, GARCH variants, transformers). The sample size and feature dimensionality do not justify the representational capacity of these models, and the risk of overfitting is disproportionately high.
- A production-grade backtesting engine with transaction costs and slippage. The backtest evaluates forecast quality via realized risk metrics, not net profitability. Adding cost assumptions introduces confounding parameters.
- Directional price forecasting or market timing. This is a risk model, not a return prediction model.
- Multi-horizon volatility term structure modeling.
- Live trading infrastructure or API integration.

---

## 5. Assumptions

The following assumptions define the project's context. They are stated boundaries, not hidden limitations.

1. **Volatility Clustering:** Volatility regimes persist. Relationships learned in one regime may hold in the near term but are subject to decay. The purged walk-forward validation framework is specifically designed to test robustness under this assumption.

2. **Liquidity:** The selected large-cap stocks are sufficiently liquid that the position sizes implied by dynamic sizing do not materially impact market prices.

3. **Target Volatility is Exogenous:** A constant target volatility of 15% annualized is chosen by the hypothetical portfolio manager. The model's role is to help achieve this target, not to determine what the target should be.

4. **Stationarity of Feature Relationships:** First-order relationships between market-context features (e.g., VIX level) and single-stock volatility are assumed stable enough for a machine learning model to capture. Structural breaks in these relationships represent model risk that is acknowledged but not explicitly modeled.

5. **No Transaction Costs:** The backtest simulation does not include trading costs or market impact. This is a deliberate simplification. The project's focus is on the quality of the volatility forecast, not on the net profitability of a trading strategy. Adding transaction costs would obscure the signal being evaluated.

6. **Univariate Modeling:** Each stock is modeled independently. Cross-asset dependencies are captured only through market-level features like VIX and sector-average volatility. Full covariance modeling is a separate, more complex problem.

---

## 6. Success Criteria

Project success is evaluated on three levels.

### Primary Statistical Criteria

- Out-of-sample RMSE must be strictly lower than all three baselines: rolling historical volatility, EWMA, and ridge regression.
- The Mincer-Zarnowitz regression must fail to reject the joint null hypothesis (α=0, β=1) at the p > 0.05 level, indicating no statistically significant bias in the forecasts.

### Primary Financial Criteria

- A dynamic position-sizing strategy using the model forecast must achieve:
  - A higher out-of-sample Sharpe ratio than a static sizing strategy over the same period
  - A lower peak-to-trough maximum drawdown
- The realized portfolio volatility under dynamic sizing should be closer to the 15% target than under static sizing, measured as absolute deviation from target.

### Qualitative Criteria

- The volatility cone visualization must show that forecast errors fall within expected dispersion bands and converge appropriately as the horizon extends.
- Any observed failure modes (e.g., underperformance during specific market regimes) must be documented and discussed.

---

## 7. Baseline Solutions

A ladder of evidence is built before considering any advanced approach. Complexity is justified only when it demonstrably outperforms simpler alternatives on pre-specified criteria.

### Baseline 1: Rolling Historical Volatility

**Description:** The forecast for 5-day forward volatility is the annualized standard deviation of daily log returns over the prior 21 trading days.

**Rationale:** This is the simplest model that embodies a defensible financial intuition: near-term future volatility resembles recent past volatility. It represents the null hypothesis that all relevant information about near-term volatility is contained in the recent historical record. It requires no parameter estimation, no training, and no risk of overfitting, making it the unambiguous lower performance bound.

**Expected Performance:**
- Out-of-sample RMSE approximately equal to the cross-sectional average of realized volatility standard deviation over the test period.
- Mincer-Zarnowitz regression will likely reject the joint null (β significantly less than 1), indicating systematic under-reaction to volatility changes.
- Dynamic sizing will reduce but not eliminate target volatility violations. Expect realized volatility to deviate from the 15% target by ±4-6% annualized in normal markets.
- Tail errors (95th percentile of absolute forecast error) will be large, as the 21-day window responds slowly to volatility spikes.

**What It Validates:** Establishes that forward volatility is not simply the backward-looking average. If any model cannot beat this baseline in RMSE, it is worse than doing nothing. Also validates the data pipeline, train/test split integrity, and target variable construction.

---

### Baseline 2: Exponentially Weighted Moving Average (EWMA)

**Description:** The forecast is a weighted moving average of past squared returns, with weights decaying exponentially at rate λ=0.94 (the RiskMetrics standard). This gives a half-life of approximately 11 trading days.

**Rationale:** This baseline introduces the first structured assumption: recent observations should carry more weight than distant ones when forecasting near-term volatility. It directly addresses the equal-weighting limitation of Baseline 1 while adding only a single fixed parameter. It remains a univariate model requiring no training and no cross-sectional information. It is the minimal model that explicitly encodes volatility clustering, one of the most well-established empirical regularities in financial time series.

**Expected Performance:**
- RMSE reduction of 5-15% relative to Baseline 1, driven by faster response to recent volatility shocks.
- Mincer-Zarnowitz β coefficient closer to 1 than Baseline 1 but still likely biased (β ≈ 0.70-0.85), as the fixed decay rate cannot adapt to different volatility regimes.
- Dynamic sizing will show lower peak-to-trough drawdown than Baseline 1, as the model adjusts positions more quickly when volatility spikes.
- Tail errors reduced but still substantial, as the model has no forward-looking information.

**What It Validates:** Tests whether responsiveness to recent shocks matters more than a longer memory. If EWMA does not outperform rolling historical volatility, it suggests either the 21-day window was already near-optimal or the 5-day horizon requires a different decay structure. Validates that the basic time-series properties of the data are being correctly captured.

---

### Baseline 3: Ridge-Regularized Linear Model with Curated Features

**Description:** A ridge regression (L2-penalized linear model) trained on a curated feature set: lagged EWMA volatilities at 5, 10, 21, and 63-day windows; Parkinson range-based volatility estimate; the stock's Average True Range (ATR) percentile rank over the past 63 days; the VIX closing level; and the sector-average EWMA volatility.

**Rationale:** This is the first baseline that learns from multiple data sources and introduces forward-looking market-context features. It tests the hypothesis that volatility is a linear function of a small set of well-chosen predictors. Ridge regularization is used instead of OLS because feature collinearity is expected (e.g., different volatility windows are correlated), and the L2 penalty provides numerical stability and mild shrinkage. This baseline is genuinely competitive. If a linear model with good features performs comparably to a complex non-linear model, the linear model is preferred on grounds of interpretability, simplicity, and robustness.

**Expected Performance:**
- RMSE reduction of 10-25% relative to Baseline 2. The largest improvements are expected from VIX and sector-volatility features providing forward-looking information.
- Mincer-Zarnowitz β expected in the 0.80-0.95 range. The joint null may still be rejectable at p<0.05, but the F-statistic should be substantially lower than Baselines 1 and 2.
- Dynamic sizing will show realized volatility consistently closer to the 15% target (within ±2-3% annualized) and a higher Sharpe ratio than static sizing.
- Tail errors reduced by the inclusion of VIX as a leading indicator; the model will partially anticipate volatility spikes before they fully appear in historical returns.
- Ridge coefficients will provide interpretable feature importance: VIX and sector-volatility coefficients should be positive and statistically significant.

**What It Validates:** This is the most important baseline. It tests whether the relationship between market-context features and future volatility can be adequately captured by a linear model. It provides the evidentiary foundation for any decision to introduce non-linearity. It also directly validates that the feature set contains predictive signal beyond what is available from a stock's own history.

---

## 8. Advanced Solution (Conditional)

### Averaged Ensemble of Random Forest and LightGBM

**Description:** A simple average of two structurally different tree-based models. Random Forest uses bootstrap aggregation with random feature subsets to reduce variance. LightGBM uses gradient boosting with leaf-wise tree growth to reduce bias. The ensemble prediction is the unweighted arithmetic mean of both models' outputs.

**Rationale for Consideration:** Tree-based models can capture non-linear and interaction effects that a linear model cannot. A spike in VIX may matter more when sector volatility is already elevated. Negative returns may have a disproportionately large impact on future volatility compared to positive returns of equal magnitude (the leverage effect). These are documented empirical phenomena that a linear model cannot represent without explicit feature engineering. The ensemble of two algorithmically different approaches may reduce model variance compared to either model alone. Averaging bagging and boosting predictions can produce forecasts that are more stable out-of-sample, which is valuable for a risk model where erratic forecasts would cause unnecessary position turnover.

**Trigger Condition for Adoption:** The ensemble is only adopted if both of the following conditions are met:
1. Ramsey's RESET test applied to the Ridge regression residuals rejects the null hypothesis of correct linear specification at p < 0.05. This provides statistical evidence that non-linear relationships exist in the data.
2. The Ridge regression baseline achieves a Mincer-Zarnowitz β less than 0.90, indicating that the linear model's calibration is meaningfully imperfect.

If both conditions are not met, the Ridge regression baseline is the preferred model. This finding is itself informative: it demonstrates that a well-specified linear model is sufficient for the problem at hand.

**Trade-offs:**
- Loss of coefficient-level interpretability. SHAP values and feature importance metrics provide attribution but do not support the same statistical inference as linear regression coefficients.
- Increased risk of overfitting. The purged walk-forward cross-validation framework is essential to mitigate this.
- Higher computational cost, though still manageable for this dataset size on standard hardware.

**Evaluation Protocol (if trigger conditions are met):**
1. **Mincer-Zarnowitz comparison:** The ensemble must show an F-statistic lower than Ridge regression and β meaningfully closer to 1 (difference > 0.05).
2. **Tail error comparison:** The ensemble must reduce the 95th percentile of absolute forecast error by at least 5% relative to Ridge regression.
3. **Diebold-Mariano test:** The RMSE reduction must be statistically significant using the Diebold-Mariano test with autocorrelation-consistent standard errors.
4. **Economic significance:** The ensemble-driven sizing strategy must show either a higher Sharpe ratio (difference > 0.1) or a meaningfully lower maximum drawdown (>10% relative reduction).
5. **Stability check:** The standard deviation of position size changes under the ensemble must not exceed that of the Ridge model. If the ensemble produces more volatile position changes, the expected stability benefit has not materialized.

If the ensemble does not clearly outperform on these criteria, the Ridge regression baseline is the final model.

---

## 9. Limitations

The following limitations are acknowledged. Identifying them does not weaken the project; it demonstrates understanding of what the model can and cannot do.

1. **Single-Horizon Forecasting:** Only 5-day ahead volatility is modeled. The term structure of volatility across horizons is not addressed.

2. **Univariate Scope:** Each stock is modeled independently. Dependencies between stocks are captured only through aggregate features like VIX and sector volatility. A full covariance model is a separate undertaking.

3. **No Explicit Regime-Switching Mechanism:** The model does not explicitly model discrete volatility regimes. It relies on features like VIX to provide continuous regime context. A sudden, unprecedented regime shift (e.g., a market event unlike anything in the training window) would likely produce a poor forecast.

4. **Simplified Backtest:** The position sizing simulation does not include transaction costs, slippage, or market impact. The backtest is a measurement tool for forecast quality, not a profitability claim.

5. **Data Frequency:** Daily data is sufficient for 5-day horizon forecasting but cannot capture intraday volatility dynamics. High-frequency data would provide a richer signal but is intentionally excluded to maintain manageable scope.

6. **No Causal Claims:** The model identifies predictive relationships, not causal ones. A feature being important in the model does not imply it causes changes in volatility.

---

## 10. Resume and Portfolio Signal

To a hiring manager reviewing a Finance Data Scientist intern candidate, this project communicates several signals of preparedness for quantitative finance work:

### 1. Risk-First Framing

Most entry-level portfolio projects attempt to predict price direction. This project frames volatility forecasting—uncertainty quantification—as the primary problem. It demonstrates understanding that risk management is at least as important as return forecasting in systematic investing. This is a meaningful distinction at the early-career level.

### 2. Statistical Rigor Over Model Complexity

The emphasis on the Mincer-Zarnowitz regression for unbiasedness testing, the volatility cone for calibration assessment, and the baseline ladder for complexity justification demonstrates familiarity with econometric evaluation standards. This is distinct from projects that report only RMSE or accuracy and describe the model as sufficient.

### 3. End-to-End Application Thinking

The project does not stop at model predictions. It connects the forecast to a concrete position-sizing rule and evaluates the rule's performance. This demonstrates the ability to translate model output into a business decision, which is the distinguishing contribution of a data scientist in a finance context.

### 4. Methodological Discipline

By presenting a baseline ladder and explicit trigger conditions for adopting the advanced model, the project communicates a willingness to let evidence drive methodology. A hiring manager values this judgment more than a forced application of a complex algorithm.

### 5. Focused Scope

The deliberate exclusions (no portfolio optimization, no deep learning, no live trading) indicate understanding of the difference between a research project and a production system. Scope discipline is a practical skill for an intern who will need to deliver results in a limited timeframe.

---

## 11. Distinction from Common Entry-Level Projects

Many entry-level quantitative finance projects follow a predictable pattern: download stock prices, train a classifier to predict up/down, report accuracy, and claim a profitable strategy. These projects often ignore transaction costs, use random train/test splits on time-series data, and make no attempt at statistical evaluation of forecasts.

This project takes a different approach:

| Common Entry-Level Project | This Project |
|---|---|
| Predicts price direction | Predicts volatility magnitude |
| Uses classification accuracy as the primary metric | Uses RMSE, Mincer-Zarnowitz regression, and volatility cone calibration |
| Randomly shuffles time-series data for cross-validation | Uses purged walk-forward temporal cross-validation |
| No connection to a concrete risk management decision | Directly drives a position-sizing rule with measurable economic impact |
| Complexity without justification | Baseline ladder with explicit criteria for advancing to a more complex model |

The distinction is not primarily in technical difficulty. It is in the problem formulation, the rigor of the evaluation, and the clarity of the practical application.

---

## 12. References

- Andersen, T. G., & Bollerslev, T. (1998). Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts. *International Economic Review*, 39(4), 885–905.
- Mincer, J. A., & Zarnowitz, V. (1969). The Evaluation of Economic Forecasts. In J. A. Mincer (Ed.), *Economic Forecasts and Expectations: Analysis of Forecasting Behavior and Performance*. NBER.
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. *Journal of Business*, 53(1), 61–65.
- Poon, S.-H., & Granger, C. W. J. (2003). Forecasting Volatility in Financial Markets: A Review. *Journal of Economic Literature*, 41(2), 478–539.
- RiskMetrics Group. (1996). *RiskMetrics — Technical Document* (4th ed.). J.P. Morgan/Reuters.