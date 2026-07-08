"""
Feature engineering module for volatility forecasting.

This module computes all features (Baseline 1, 2, 3, and Advanced) in a single pass.
Features are filtered based on config.FEATURE_SET for efficient experimentation.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src import config


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for all tickers.

    Args:
        prices: Multi-index DataFrame with 'Close' prices per ticker.

    Returns:
        DataFrame with log returns, one column per ticker, aligned by date.
    """
    if isinstance(prices.columns, pd.MultiIndex):
        # Extract Close prices for all tickers
        close_prices = prices.xs('Close', axis=1, level=1)
    else:
        # Single ticker case
        close_prices = prices

    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns


def compute_forward_volatility(
    returns: pd.DataFrame,
    horizon: int = 5,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute forward annualized realized volatility over a given horizon.

    For each day t, calculates volatility from log returns over days t+1 to t+horizon.

    Args:
        returns: DataFrame of log returns (dates x tickers).
        horizon: Number of forward days to use.
        annualize: If True, annualize by sqrt(252).

    Returns:
        DataFrame of forward volatilities (dates x tickers), aligned so
        row t contains the target for day t.
    """
    # Ensure returns are sorted by date
    returns = returns.sort_index()

    # Compute rolling standard deviation over future window (forward-looking)
    # Using rolling with shift to look ahead
    forward_vol = returns.rolling(window=horizon).std().shift(-horizon)

    if annualize:
        forward_vol = forward_vol * np.sqrt(config.ANNUALIZATION_FACTOR)

    return forward_vol


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute rolling historical volatility over a given window.

    Args:
        returns: DataFrame of log returns (dates x tickers).
        window: Rolling window size in trading days.
        annualize: If True, annualize by sqrt(252).

    Returns:
        DataFrame of rolling volatilities (dates x tickers).
    """
    rolling_vol = returns.rolling(window=window).std()

    if annualize:
        rolling_vol = rolling_vol * np.sqrt(config.ANNUALIZATION_FACTOR)

    return rolling_vol


def compute_ewma_volatility(
    returns: pd.DataFrame,
    lambda_: float = 0.94,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute EWMA volatility using RiskMetrics lambda parameter.

    Variance at time t = λ * variance_{t-1} + (1-λ) * return_{t-1}^2

    Args:
        returns: DataFrame of log returns (dates x tickers).
        lambda_: Decay factor (0.94 is RiskMetrics standard).
        annualize: If True, annualize by sqrt(252).

    Returns:
        DataFrame of EWMA volatilities (dates x tickers).
    """
    squared_returns = returns ** 2
    ewma_var = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    # Initialize with 21-day average of squared returns (more stable)
    init_var = squared_returns.iloc[:21].mean().values

    # Set initial variance to the 21-day average
    ewma_var.iloc[0] = init_var

    # Compute EWMA recursively
    for i in range(1, len(returns)):
        ewma_var.iloc[i] = lambda_ * ewma_var.iloc[i-1].values + (1 - lambda_) * squared_returns.iloc[i-1].values

    ewma_vol = np.sqrt(ewma_var)

    if annualize:
        ewma_vol = ewma_vol * np.sqrt(config.ANNUALIZATION_FACTOR)

    return ewma_vol


def compute_parkinson_volatility(
    prices: pd.DataFrame,
    window: int = 21,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute Parkinson volatility estimator using high-low range.

    Parkinson (1980): σ_parkinson = sqrt( (1/(4*log(2))) * mean(log(High/Low)^2) )

    Args:
        prices: Multi-index DataFrame with 'High' and 'Low' prices per ticker.
        window: Rolling window size.
        annualize: If True, annualize by sqrt(252).

    Returns:
        DataFrame of Parkinson volatilities (dates x tickers).
    """
    if not isinstance(prices.columns, pd.MultiIndex):
        warnings.warn("Parkinson volatility requires multi-index columns with 'High' and 'Low'")
        return pd.DataFrame(index=prices.index)

    # Extract High and Low for all tickers
    high = prices.xs('High', axis=1, level=1)
    low = prices.xs('Low', axis=1, level=1)

    # Compute log(High/Low)^2
    ratio = np.log(high / low) ** 2

    # Rolling mean of squared log ratios
    parkinson_var = ratio.rolling(window=window).mean() / (4 * np.log(2))

    # Square root for volatility
    parkinson_vol = np.sqrt(parkinson_var)

    if annualize:
        parkinson_vol = parkinson_vol * np.sqrt(config.ANNUALIZATION_FACTOR)

    return parkinson_vol


def compute_atr_percentile_rank(
    prices: pd.DataFrame,
    atr_window: int = 14,
    rank_window: int = 63
) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) and its percentile rank.

    ATR measures volatility based on daily range (High - Low) and gaps.

    Args:
        prices: Multi-index DataFrame with 'High', 'Low', 'Close' prices.
        atr_window: Window for ATR calculation.
        rank_window: Window for percentile rank calculation.

    Returns:
        DataFrame of ATR percentile ranks (dates x tickers).
    """
    if not isinstance(prices.columns, pd.MultiIndex):
        warnings.warn("ATR percentile requires multi-index columns")
        return pd.DataFrame(index=prices.index)

    high = prices.xs('High', axis=1, level=1)
    low = prices.xs('Low', axis=1, level=1)
    close = prices.xs('Close', axis=1, level=1)

    # Calculate True Range
    high_low = high - low
    high_close_prev = (high - close.shift(1)).abs()
    low_close_prev = (low - close.shift(1)).abs()

    true_range = pd.DataFrame({
        'hl': high_low,
        'hc': high_close_prev,
        'lc': low_close_prev
    }, index=high.index)

    # True Range is max of the three
    true_range_max = true_range.max(axis=1, level=0)

    # ATR is EMA or SMA of True Range
    atr = true_range_max.rolling(window=atr_window).mean()

    # Percentile rank over rank_window
    percentile_rank = atr.rolling(window=rank_window).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )

    return percentile_rank


def compute_vix_features(vix: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute VIX-derived features.

    Args:
        vix: DataFrame with VIX OHLCV data.

    Returns:
        Dictionary of VIX features (all as pd.Series with date index).
    """
    if vix.empty:
        warnings.warn("VIX data is empty. VIX features will be None.")
        return {}

    # Ensure we have Close prices
    if 'Close' not in vix.columns:
        warnings.warn("VIX data missing 'Close' column")
        return {}

    vix_close = vix['Close']
    # Ensure it's a 1D Series
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]

    features = {
        'vix_level': vix_close,
        'vix_change_5d': vix_close.pct_change(periods=5),
        'vix_change_21d': vix_close.pct_change(periods=21),
        'vix_21d_avg': vix_close.rolling(window=21).mean(),
        'vix_term_structure': vix_close / vix_close.rolling(window=21).mean(),
    }

    return features


def compute_advanced_features(
    returns: pd.DataFrame,
    rolling_vol_21: pd.DataFrame,
    rolling_vol_63: pd.DataFrame,
    rolling_vol_252: pd.DataFrame,
    ewma_vol_94: pd.DataFrame,
    vix_level: pd.Series,
    sector_ewma: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute advanced interaction and regime features.

    These capture non-linear relationships and regime context.

    Args:
        returns: Log returns DataFrame.
        rolling_vol_21: 21-day rolling volatility.
        rolling_vol_63: 63-day rolling volatility.
        rolling_vol_252: 252-day rolling volatility.
        ewma_vol_94: EWMA volatility (λ=0.94).
        vix_level: VIX level series (1D).
        sector_ewma: Sector-average EWMA volatility (optional).

    Returns:
        Dictionary of advanced features (DataFrames with dates x tickers).
    """
    features = {}

    # 1. Volatility regime: current vol relative to long-term average
    features['vol_regime'] = rolling_vol_21 / rolling_vol_252

    # 2. VIX × Rolling Vol: interaction effect
    # Ensure vix_level is a 1D Series
    if isinstance(vix_level, pd.DataFrame):
        vix_level = vix_level.iloc[:, 0]
    elif isinstance(vix_level, np.ndarray) and vix_level.ndim == 2:
        vix_level = vix_level.flatten()
    
    # Broadcast VIX to all tickers
    vix_broadcast = pd.DataFrame(
        {ticker: vix_level.values for ticker in returns.columns},
        index=returns.index
    )
    features['vix_times_rolling_vol'] = vix_broadcast * rolling_vol_21

    # 3. Leverage effect: negative returns amplify volatility
    negative_returns = (returns < 0).astype(float)
    features['leverage_effect'] = negative_returns * rolling_vol_21

    # 4. Volatility of volatility: stability of the volatility process
    features['vol_of_vol'] = rolling_vol_21.rolling(window=63).std()

    # 5. Return reversal: large price moves (absolute return over 5 days)
    features['return_reversal_5d'] = returns.rolling(window=5).mean().abs()

    # 6. Market stress: VIX relative to normal level (20)
    features['market_stress'] = (vix_level - 20) / 20
    features['market_stress'] = pd.DataFrame(
        {ticker: features['market_stress'].values for ticker in returns.columns},
        index=returns.index
    )

    # 7. Sector relative volatility (if sector EWMA provided)
    if sector_ewma is not None:
        features['sector_relative_vol'] = ewma_vol_94 / sector_ewma

    return features

def compute_sector_ewma(
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    lambda_: float = 0.94
) -> pd.DataFrame:
    """
    Compute sector-average EWMA volatility.

    For each ticker, calculates the average EWMA volatility of all stocks
    in the same sector (excluding the ticker itself).

    Args:
        returns: Log returns DataFrame (dates x tickers).
        sector_map: Dictionary mapping ticker -> sector name.
        lambda_: EWMA decay factor.

    Returns:
        DataFrame of sector-average EWMA volatilities (dates x tickers).
    """
    # Compute EWMA variance for each ticker
    squared_returns = returns ** 2
    ewma_var = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    ewma_var.iloc[0] = squared_returns.iloc[0].values

    for i in range(1, len(returns)):
        ewma_var.iloc[i] = lambda_ * ewma_var.iloc[i-1].values + (1 - lambda_) * squared_returns.iloc[i-1].values

    ewma_vol = np.sqrt(ewma_var) * np.sqrt(config.ANNUALIZATION_FACTOR)

    # Group by sector and compute average
    sector_ewma = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for ticker in returns.columns:
        sector = sector_map.get(ticker)
        if sector is None:
            sector_ewma[ticker] = ewma_vol[ticker]
            continue

        # Get other tickers in same sector
        same_sector = [t for t, s in sector_map.items() if s == sector and t != ticker]
        if same_sector:
            sector_ewma[ticker] = ewma_vol[same_sector].mean(axis=1)
        else:
            sector_ewma[ticker] = ewma_vol[ticker]

    return sector_ewma


def assemble_feature_matrix(
    prices: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble complete feature matrix with all features.
    """
    # 1. Compute log returns
    returns = compute_log_returns(prices)

    # 2. Compute target
    target = compute_forward_volatility(returns, horizon=config.FORECAST_HORIZON)

    # 3. Compute baseline features
    rolling_vol_21 = compute_rolling_volatility(returns, 21)
    rolling_vol_63 = compute_rolling_volatility(returns, 63)
    rolling_vol_252 = compute_rolling_volatility(returns, 252)

    ewma_vol_94 = compute_ewma_volatility(returns, 0.94)
    ewma_vol_90 = compute_ewma_volatility(returns, 0.90)
    ewma_vol_97 = compute_ewma_volatility(returns, 0.97)

    parkinson_vol = compute_parkinson_volatility(prices, window=21)

    # 4. Compute VIX features
    vix_features = compute_vix_features(vix)
    vix_level = vix_features.get('vix_level', pd.Series(index=returns.index, dtype=float))

    # Ensure vix_level is 1D Series
    if isinstance(vix_level, pd.DataFrame):
        vix_level = vix_level.iloc[:, 0]
    elif isinstance(vix_level, np.ndarray) and vix_level.ndim == 2:
        vix_level = vix_level.flatten()

    # Handle vix_change_5d - fill first 5 days with 0 (no change)
    vix_change_5d = vix_features.get('vix_change_5d', pd.Series(index=returns.index, dtype=float))
    if isinstance(vix_change_5d, pd.DataFrame):
        vix_change_5d = vix_change_5d.iloc[:, 0]
    vix_change_5d = vix_change_5d.fillna(0)

    # 5. Compute sector EWMA
    sector_ewma = compute_sector_ewma(returns, config.SECTOR_MAP)

    # 6. Compute advanced features
    advanced_features = compute_advanced_features(
        returns=returns,
        rolling_vol_21=rolling_vol_21,
        rolling_vol_63=rolling_vol_63,
        rolling_vol_252=rolling_vol_252,
        ewma_vol_94=ewma_vol_94,
        vix_level=vix_level,
        sector_ewma=sector_ewma,
    )

    # 7. Assemble all features into a single DataFrame (long format)
    all_features = {}

    # Baseline 1 features
    all_features['rolling_vol_21'] = rolling_vol_21
    all_features['rolling_vol_63'] = rolling_vol_63
    all_features['rolling_vol_252'] = rolling_vol_252

    # Baseline 2 features
    all_features['ewma_vol_94'] = ewma_vol_94
    all_features['ewma_vol_90'] = ewma_vol_90
    all_features['ewma_vol_97'] = ewma_vol_97

    # Baseline 3 features (additional)
    all_features['parkinson_vol_21'] = parkinson_vol
    all_features['vix_level'] = pd.DataFrame(
        {ticker: vix_level.values for ticker in returns.columns},
        index=returns.index
    )
    all_features['vix_change_5d'] = pd.DataFrame(
        {ticker: vix_change_5d.values for ticker in returns.columns},
        index=returns.index
    )
    all_features['sector_ewma'] = sector_ewma

    # Advanced features
    for name, feature_df in advanced_features.items():
        all_features[name] = feature_df

    # 8. Convert to long format (date, ticker, target, features)
    feature_list = []
    for ticker in returns.columns:
        ticker_data = pd.DataFrame({
            'date': returns.index,
            'ticker': ticker,
            'target': target[ticker].values,
        })

        # Add all features for this ticker
        for feature_name, feature_df in all_features.items():
            if ticker in feature_df.columns:
                ticker_data[feature_name] = feature_df[ticker].values
            else:
                # Fallback: use mean or broadcast if feature is missing
                if isinstance(feature_df, pd.Series):
                    ticker_data[feature_name] = feature_df.values
                else:
                    ticker_data[feature_name] = np.nan

        feature_list.append(ticker_data)

    feature_df = pd.concat(feature_list, ignore_index=True)

    # 9. Sort by ticker and date
    feature_df = feature_df.sort_values(['ticker', 'date'])

    # 10. Forward fill features within each ticker
    feature_cols = [c for c in feature_df.columns if c not in ['date', 'ticker', 'target']]
    
    # Group by ticker and forward fill each feature
    for col in feature_cols:
        feature_df[col] = feature_df.groupby('ticker')[col].ffill()

    # 11. For remaining NaN values, use global median (not per ticker, to avoid empty groups)
    for col in feature_cols:
        if feature_df[col].isna().any():
            median_val = feature_df[col].median()
            if pd.isna(median_val):
                median_val = 0
            feature_df[col] = feature_df[col].fillna(median_val)

    # 12. Drop rows where target is NaN (target cannot be imputed)
    initial_len = len(feature_df)
    feature_df = feature_df.dropna(subset=['target'])
    target_dropped = initial_len - len(feature_df)
    if target_dropped > 0:
        warnings.warn(f"Dropped {target_dropped} rows with NaN target (end of series)")

    # 13. Final check - drop any remaining NaN
    before_final = len(feature_df)
    feature_df = feature_df.dropna()
    final_dropped = before_final - len(feature_df)
    if final_dropped > 0:
        warnings.warn(f"Dropped {final_dropped} rows with remaining NaN values")

    # 14. Final validation
    if len(feature_df) == 0:
        raise ValueError(
            f"Feature matrix is empty after processing. "
            f"Initial rows: {initial_len}, "
            f"Target NaN rows: {target_dropped}, "
            f"Final drop: {final_dropped}. "
            f"Check data quality and feature calculations."
        )

    print(f"Final feature matrix shape: {feature_df.shape}")
    print(f"Date range: {feature_df['date'].min()} to {feature_df['date'].max()}")
    print(f"Unique tickers: {feature_df['ticker'].nunique()}")

    return feature_df


def get_feature_list(feature_set: str) -> List[str]:
    """
    Get list of feature column names for a given feature set.

    Args:
        feature_set: 'baseline_1', 'baseline_2', 'baseline_3', 'advanced'

    Returns:
        List of feature column names.
    """
    all_features = [
        'rolling_vol_21', 'rolling_vol_63', 'rolling_vol_252',
        'ewma_vol_94', 'ewma_vol_90', 'ewma_vol_97',
        'parkinson_vol_21', 'vix_level', 'vix_change_5d', 'sector_ewma',
        'vol_regime', 'vix_times_rolling_vol', 'leverage_effect',
        'vol_of_vol', 'return_reversal_5d', 'market_stress', 'sector_relative_vol'
    ]

    if feature_set == 'baseline_1':
        # Keep all 3 rolling volatility features (no redundancy >0.8)
        return ['rolling_vol_21', 'rolling_vol_63', 'rolling_vol_252']
    
    elif feature_set == 'baseline_2':
        # Remove redundant EWMA features (perfectly correlated)
        # Keep only the RiskMetrics standard (λ=0.94)
        return ['ewma_vol_94']
    
    elif feature_set == 'baseline_3':
        # Remove rolling_vol_21 (redundant with parkinson_vol_21 at 0.927)
        # Keep parkinson_vol_21 (captures intraday high-low range)
        return ['parkinson_vol_21', 'ewma_vol_94', 'vix_level', 'vix_change_5d', 'rolling_vol_63']
    
    elif feature_set == 'advanced':
        # Remove rolling_vol_21 (redundant with parkinson_vol_21 at 0.927)
        # Remove market_stress (perfectly correlated with vix_level at 1.000)
        return [
            'parkinson_vol_21', 'ewma_vol_94',
            'vix_level', 'vix_change_5d', 'rolling_vol_63',
            'vix_times_rolling_vol', 'vol_regime', 'leverage_effect',
            'vol_of_vol', 'return_reversal_5d', 'sector_relative_vol'
        ]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    


    
def filter_features(
    feature_df: pd.DataFrame,
    feature_set: str
) -> pd.DataFrame:
    """
    Filter feature matrix to a specific feature set.

    Args:
        feature_df: Full feature matrix from assemble_feature_matrix().
        feature_set: 'baseline_1', 'baseline_2', 'baseline_3', 'advanced'

    Returns:
        Filtered DataFrame with only selected features.
    """
    if feature_set not in ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    selected_features = get_feature_list(feature_set)

    # Keep date, ticker, target, and selected features
    keep_cols = ['date', 'ticker', 'target'] + selected_features
    return feature_df[keep_cols]


def save_processed_data(feature_df: pd.DataFrame, path: str) -> None:
    """
    Save processed feature matrix to disk.

    Args:
        feature_df: Feature matrix DataFrame.
        path: File path (should end with .parquet or .csv).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith('.parquet'):
        feature_df.to_parquet(path, index=False)
    elif path.endswith('.csv'):
        feature_df.to_csv(path, index=False)
    else:
        raise ValueError("Path must end with .parquet or .csv")


def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load processed feature matrix from disk.

    Args:
        path: File path (.parquet or .csv).

    Returns:
        Feature matrix DataFrame.
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path, parse_dates=['date'])
    else:
        raise ValueError("Path must end with .parquet or .csv")


def build_feature_matrix(
    prices: pd.DataFrame,
    vix: pd.DataFrame,
    feature_set: Optional[str] = None,
    save_path: Optional[str] = None,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Main entry point for feature engineering.

    Builds the full feature matrix, optionally filters to a feature set,
    and optionally saves to disk.

    Args:
        prices: Multi-index DataFrame with OHLCV data.
        vix: DataFrame with VIX data.
        feature_set: If provided, filter to this feature set.
        save_path: If provided, save to this path.
        force_rebuild: If True, rebuild even if saved file exists.

    Returns:
        Feature matrix DataFrame (filtered if feature_set provided).
    """
    # Try to load from cache if available and not forcing rebuild
    if save_path and not force_rebuild and os.path.exists(save_path):
        print(f"Loading cached feature matrix from {save_path}")
        feature_df = load_processed_data(save_path)

        if feature_set:
            feature_df = filter_features(feature_df, feature_set)

        return feature_df

    # Build from scratch
    print("Building feature matrix from scratch...")
    feature_df = assemble_feature_matrix(prices, vix)

    # Save if path provided
    if save_path:
        print(f"Saving full feature matrix to {save_path}")
        save_processed_data(feature_df, save_path)

    # Filter if feature_set provided
    if feature_set:
        print(f"Filtering to feature set: {feature_set}")
        feature_df = filter_features(feature_df, feature_set)

    return feature_df


# Quick test function
if __name__ == "__main__":
    print("Feature engineering module loaded successfully.")
    print(f"Available feature sets: {config.FEATURE_SET}")
    print(f"Baseline 1 features: {get_feature_list('baseline_1')}")
    print(f"Baseline 2 features: {get_feature_list('baseline_2')}")
    print(f"Baseline 3 features: {get_feature_list('baseline_3')}")
    print(f"Advanced features: {get_feature_list('advanced')}")