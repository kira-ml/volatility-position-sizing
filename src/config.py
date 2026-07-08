"""
Configuration module for volatility forecasting project.

Centralizes all configuration constants used across the pipeline.
All settings can be overridden via environment variables or command-line
arguments in the main pipeline script.
"""

from typing import Dict, List


# =============================================================================
# Data Configuration
# =============================================================================

# Primary stock tickers (liquid US large-cap stocks)
TICKERS: List[str] = [
    'AAPL',   # Apple Inc.
    'MSFT',   # Microsoft Corporation
    'GOOGL',  # Alphabet Inc. (Class A)
    'AMZN',   # Amazon.com Inc.
    'META',   # Meta Platforms Inc.
    'JPM',    # JPMorgan Chase & Co.
    'XOM',    # Exxon Mobil Corporation
    'JNJ',    # Johnson & Johnson
    'WMT',    # Walmart Inc.
    'TSLA',   # Tesla Inc.
]

# Date range for data collection (approximately 5 years)
START_DATE: str = '2020-01-01'
END_DATE: str = '2024-12-31'

# Sector mapping for sector-average volatility calculations
SECTOR_MAP: Dict[str, str] = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'AMZN': 'Consumer',
    'META': 'Technology',
    'JPM': 'Financial',
    'XOM': 'Energy',
    'JNJ': 'Healthcare',
    'WMT': 'Consumer',
    'TSLA': 'Consumer',
}


# =============================================================================
# Feature Engineering Configuration
# =============================================================================

# Target horizon (days)
FORECAST_HORIZON: int = 5

# Volatility calculation constants
ANNUALIZATION_FACTOR: float = 252.0  # Trading days per year

# Rolling window lengths for historical volatility
VOL_WINDOWS: List[int] = [5, 10, 21, 63]

# EWMA decay factor (RiskMetrics standard)
EWMA_LAMBDA: float = 0.94

# Parkinson volatility window (trading days)
PARKINSON_WINDOW: int = 21

# ATR and percentile rank windows
ATR_WINDOW: int = 14
ATR_RANK_WINDOW: int = 63

# Sector EWMA window
SECTOR_EWMA_WINDOW: int = 21


# =============================================================================
# Model Configuration
# =============================================================================

# Ridge regression hyperparameters
RIDGE_ALPHA: float = 1.0  # L2 regularization strength

# Train/test split ratio (80% train, 20% test)
TEST_SPLIT_RATIO: float = 0.2

# Random seed for reproducibility
RANDOM_SEED: int = 42


# =============================================================================
# Backtest Configuration
# =============================================================================

# Target annualized volatility (15%)
TARGET_VOLATILITY: float = 0.15

# Maximum leverage cap (1.0 = no leverage)
MAX_LEVERAGE: float = 1.0

# Risk-free rate for Sharpe ratio (annualized)
RISK_FREE_RATE: float = 0.0


# =============================================================================
# Path Configuration
# =============================================================================

# Base directories
DATA_PATH: str = 'data'
RAW_DATA_PATH: str = 'data/raw'
PROCESSED_DATA_PATH: str = 'data/processed'
OUTPUTS_PATH: str = 'outputs'
FIGURES_PATH: str = 'outputs/figures'
TABLES_PATH: str = 'outputs/tables'

# File naming
PRICES_FILENAME: str = 'prices.csv'
VIX_FILENAME: str = 'vix.csv'
PROCESSED_FILENAME: str = 'feature_matrix.parquet'


# =============================================================================
# Logging Configuration
# =============================================================================

# Log level (can be 'DEBUG', 'INFO', 'WARNING', 'ERROR')
LOG_LEVEL: str = 'INFO'

# Whether to show progress bars during downloads
SHOW_PROGRESS: bool = False


# =============================================================================
# Helper Functions
# =============================================================================

def get_ticker_sector(ticker: str) -> str:
    """
    Get the sector for a given ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Sector name, or 'Unknown' if not found.
    """
    return SECTOR_MAP.get(ticker, 'Unknown')


def get_sector_tickers(sector: str) -> List[str]:
    """
    Get all tickers belonging to a sector.

    Args:
        sector: Sector name.

    Returns:
        List of ticker symbols in that sector.
    """
    return [ticker for ticker, sec in SECTOR_MAP.items() if sec == sector]


def get_unique_sectors() -> List[str]:
    """
    Get all unique sectors in the ticker universe.

    Returns:
        List of unique sector names.
    """
    return list(set(SECTOR_MAP.values()))