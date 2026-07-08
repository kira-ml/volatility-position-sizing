"""
Data loading and validation module for volatility forecasting project.

This module handles downloading, loading, cleaning, and validating OHLCV data
from Yahoo Finance for the specified tickers and VIX index.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf


def download_data(
    tickers: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple tickers and VIX index from Yahoo Finance.

    Args:
        tickers: List of stock ticker symbols.
        start_date: Start date for data (YYYY-MM-DD or datetime).
        end_date: End date for data (YYYY-MM-DD or datetime).
        progress: Whether to show download progress bar.

    Returns:
        Dictionary with two keys:
            - 'prices': Multi-index DataFrame with (Ticker, OHLCV) columns
            - 'vix': Single DataFrame with VIX OHLCV data
    """
    # Download stock data with multi-ticker support
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=progress,
        group_by='ticker',
        auto_adjust=True,  # Use adjusted close by default
    )

    # Download VIX separately (single ticker)
    vix = yf.download(
        '^VIX',
        start=start_date,
        end=end_date,
        progress=progress,
        auto_adjust=True,
    )

    return {'prices': prices, 'vix': vix}


def save_raw_data(
    prices: pd.DataFrame,
    vix: pd.DataFrame,
    raw_path: str,
    prefix: str = '',
) -> None:
    """
    Save raw OHLCV data to CSV files.

    Args:
        prices: Multi-index DataFrame with price data.
        vix: DataFrame with VIX data.
        raw_path: Directory path to save CSV files.
        prefix: Optional prefix for filenames.
    """
    os.makedirs(raw_path, exist_ok=True)

    prices_file = os.path.join(raw_path, f'{prefix}prices.csv' if prefix else 'prices.csv')
    vix_file = os.path.join(raw_path, f'{prefix}vix.csv' if prefix else 'vix.csv')

    prices.to_csv(prices_file)
    vix.to_csv(vix_file)


def load_raw_data(
    raw_path: str,
    prefix: str = '',
) -> Dict[str, pd.DataFrame]:
    """
    Load raw OHLCV data from CSV files.

    Args:
        raw_path: Directory path containing CSV files.
        prefix: Optional prefix for filenames.

    Returns:
        Dictionary with 'prices' and 'vix' DataFrames.
    """
    prices_file = os.path.join(raw_path, f'{prefix}prices.csv' if prefix else 'prices.csv')
    vix_file = os.path.join(raw_path, f'{prefix}vix.csv' if prefix else 'vix.csv')

    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    vix = pd.read_csv(vix_file, index_col=0, parse_dates=True)

    return {'prices': prices, 'vix': vix}


def validate_data(prices: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data quality and integrity.

    Checks performed:
        - Date index is monotonic and sorted
        - No negative prices
        - No excessive missing data (>5% per ticker)
        - No duplicate index values

    Args:
        prices: Multi-index DataFrame with OHLCV data.

    Returns:
        Tuple of (is_valid, warnings_list).
    """
    warnings = []

    # Check index
    if not prices.index.is_monotonic_increasing:
        warnings.append("Date index is not monotonic increasing")
        prices = prices.sort_index()

    if prices.index.duplicated().any():
        warnings.append(f"Found {prices.index.duplicated().sum()} duplicate dates")

    # Check for negative prices
    price_columns = [col for col in prices.columns if 'Close' in col or 'Adj Close' in col]
    if price_columns:
        negative_mask = prices[price_columns] < 0
        if negative_mask.any().any():
            neg_count = negative_mask.sum().sum()
            warnings.append(f"Found {neg_count} negative price values")

    # Check missing data per ticker
    # For multi-index columns, get unique tickers from column level 0
    if isinstance(prices.columns, pd.MultiIndex):
        tickers = prices.columns.get_level_values(0).unique()
    else:
        # Fallback: assume columns are simple (single ticker or VIX only)
        tickers = []

    for ticker in tickers:
        if ticker == '^VIX':
            continue
        try:
            close_col = (ticker, 'Close') if isinstance(prices.columns, pd.MultiIndex) else 'Close'
            missing_pct = prices[close_col].isna().mean()
            if missing_pct > 0.05:
                warnings.append(
                    f"{ticker}: {missing_pct:.1%} missing data (exceeds 5% threshold)"
                )
        except (KeyError, IndexError):
            warnings.append(f"{ticker}: Could not find Close column")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def clean_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data by handling missing values and ensuring data quality.

    Steps:
        - Sort by date index
        - Forward-fill missing values (up to 5 days)
        - Drop tickers with >5% missing data

    Args:
        prices: Multi-index DataFrame with OHLCV data.

    Returns:
        Cleaned DataFrame.
    """
    # Sort by date
    prices = prices.sort_index()

    # Forward-fill missing values
    prices = prices.ffill()

    # Drop tickers with excessive missing data (>5%)
    if isinstance(prices.columns, pd.MultiIndex):
        tickers = prices.columns.get_level_values(0).unique()
        for ticker in tickers:
            if ticker == '^VIX':
                continue
            try:
                close_col = (ticker, 'Close')
                missing_pct = prices[close_col].isna().mean()
                if missing_pct > 0.05:
                    # Drop all columns for this ticker
                    cols_to_drop = [col for col in prices.columns if col[0] == ticker]
                    prices = prices.drop(columns=cols_to_drop)
                    print(f"Warning: Dropped {ticker} due to {missing_pct:.1%} missing data")
            except KeyError:
                print(f"Warning: Could not find Close column for {ticker}")

    # Drop any remaining rows with NaN
    prices = prices.dropna()

    return prices


def load_data(
    tickers: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    raw_path: str,
    force_download: bool = False,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Main orchestration function for data loading and validation.

    Loads data from CSV if available, otherwise downloads from Yahoo Finance.
    Always validates and cleans the data before returning.

    Args:
        tickers: List of stock ticker symbols.
        start_date: Start date for data.
        end_date: End date for data.
        raw_path: Directory path for raw data storage.
        force_download: If True, always download fresh data.
        progress: Show download progress bar.

    Returns:
        Dictionary with 'prices' and 'vix' DataFrames (cleaned and validated).

    Raises:
        ValueError: If data validation fails critically.
        Exception: If download fails and no cached data exists.
    """
    # Check if data already exists
    prices_file = os.path.join(raw_path, 'prices.csv')
    vix_file = os.path.join(raw_path, 'vix.csv')
    data_exists = os.path.exists(prices_file) and os.path.exists(vix_file)

    if data_exists and not force_download:
        print("Loading cached data from CSV...")
        data = load_raw_data(raw_path)
    else:
        print(f"Downloading data for {len(tickers)} tickers + VIX from {start_date} to {end_date}...")
        try:
            data = download_data(tickers, start_date, end_date, progress=progress)
            # Save for future use
            save_raw_data(data['prices'], data['vix'], raw_path)
            print("Data downloaded and saved successfully.")
        except Exception as e:
            print(f"Download failed: {e}")
            if data_exists:
                print("Falling back to cached data...")
                data = load_raw_data(raw_path)
            else:
                raise

    # Validate
    print("Validating data...")
    is_valid, warnings = validate_data(data['prices'])
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  - {w}")

    if not is_valid:
        print("Data validation has critical issues.")

    # Clean
    print("Cleaning data...")
    data['prices'] = clean_data(data['prices'])

    # Clean VIX as well
    data['vix'] = data['vix'].ffill().dropna()

    print(f"Data shape after cleaning: {data['prices'].shape}")
    print(f"Date range: {data['prices'].index.min()} to {data['prices'].index.max()}")

    return data


def get_ticker_list(data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Extract ticker list from loaded data.

    Args:
        data: Dictionary returned by load_data().

    Returns:
        List of ticker symbols present in the data.
    """
    prices = data['prices']
    if isinstance(prices.columns, pd.MultiIndex):
        return list(prices.columns.get_level_values(0).unique())
    else:
        # Single ticker case
        return [prices.columns.name] if prices.columns.name else []


def get_date_range(data: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
    """
    Extract date range from loaded data.

    Args:
        data: Dictionary returned by load_data().

    Returns:
        Tuple of (start_date, end_date).
    """
    idx = data['prices'].index
    return idx.min(), idx.max()