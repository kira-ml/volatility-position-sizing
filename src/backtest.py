"""
Backtest simulation module for volatility forecasting.

This module implements:
- Static position sizing (constant allocation)
- Dynamic position sizing (target_vol / predicted_vol)
- Performance metrics (Sharpe ratio, realized volatility, max drawdown)
- Backtest comparison between static and dynamic sizing
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Daily returns.
        risk_free_rate: Annualized risk-free rate (default 0).
        annualize: If True, annualize the result.

    Returns:
        Sharpe ratio.
    """
    if len(returns) < 2:
        return np.nan

    excess_returns = returns - risk_free_rate / config.ANNUALIZATION_FACTOR
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0:
        return np.nan

    sharpe = mean_return / std_return

    if annualize:
        sharpe = sharpe * np.sqrt(config.ANNUALIZATION_FACTOR)

    return sharpe


def compute_realized_volatility(
    returns: np.ndarray,
    annualize: bool = True
) -> float:
    """
    Compute realized volatility from daily returns.

    Args:
        returns: Daily returns.
        annualize: If True, annualize the result.

    Returns:
        Realized volatility.
    """
    if len(returns) < 2:
        return np.nan

    vol = np.std(returns, ddof=1)

    if annualize:
        vol = vol * np.sqrt(config.ANNUALIZATION_FACTOR)

    return vol


def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from daily returns.

    Args:
        returns: Daily returns.

    Returns:
        Maximum drawdown as a positive percentage (e.g., 0.20 for 20%).
    """
    if len(returns) == 0:
        return np.nan

    # Compute cumulative returns
    cumulative = (1 + returns).cumprod()

    # Compute running maximum
    running_max = cumulative.expanding().max()

    # Compute drawdown
    drawdown = (cumulative - running_max) / running_max

    # Return maximum drawdown as positive percentage
    return abs(drawdown.min())


def compute_vol_deviation(
    realized_vol: float,
    target_vol: float
) -> float:
    """
    Compute absolute deviation from target volatility.

    Args:
        realized_vol: Realized volatility.
        target_vol: Target volatility.

    Returns:
        Absolute deviation.
    """
    return abs(realized_vol - target_vol)


def compute_position_scale(
    target_vol: float,
    predicted_vol: float,
    max_leverage: float = 1.0,
    min_scale: float = 0.0
) -> float:
    """
    Compute position scale based on volatility forecast.

    position_scale = target_vol / predicted_vol

    Args:
        target_vol: Target volatility (annualized).
        predicted_vol: Predicted volatility (annualized).
        max_leverage: Maximum leverage cap.
        min_scale: Minimum scale (floor).

    Returns:
        Position scale (capped and floored).
    """
    if predicted_vol <= 0:
        return max_leverage

    scale = target_vol / predicted_vol

    # Apply caps and floors
    scale = min(scale, max_leverage)
    scale = max(scale, min_scale)

    return scale


def backtest_static_sizing(
    returns: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Simulate static position sizing (constant 1x allocation).

    Args:
        returns: Daily asset returns.

    Returns:
        Dictionary with:
            - returns: Daily strategy returns
            - positions: Daily position sizes (all 1.0)
    """
    positions = np.ones_like(returns)
    strategy_returns = returns * positions

    return {
        'returns': strategy_returns,
        'positions': positions,
    }


def backtest_dynamic_sizing(
    returns: np.ndarray,
    predicted_vol: np.ndarray,
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
    min_scale: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Simulate dynamic position sizing based on volatility forecast.

    position_scale = target_vol / predicted_vol

    Args:
        returns: Daily asset returns.
        predicted_vol: Predicted volatility (annualized) for each day.
        target_vol: Target volatility (annualized).
        max_leverage: Maximum leverage cap.
        min_scale: Minimum scale (floor).

    Returns:
        Dictionary with:
            - returns: Daily strategy returns
            - positions: Daily position sizes
            - scales: Raw scales before capping
    """
    # Compute position scales
    scales = np.array([
        compute_position_scale(target_vol, vol, max_leverage, min_scale)
        for vol in predicted_vol
    ])

    # Compute strategy returns
    strategy_returns = returns * scales

    return {
        'returns': strategy_returns,
        'positions': scales,
        'scales': scales,  # Same as positions for simplicity
    }


def compute_backtest_metrics(
    returns: np.ndarray,
    target_vol: Optional[float] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute all performance metrics for a backtest.

    Args:
        returns: Daily strategy returns.
        target_vol: Target volatility for deviation calculation.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Dictionary with all metrics.
    """
    metrics = {}

    # Basic metrics
    metrics['n_days'] = len(returns)
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (1 + metrics['total_return']) ** (config.ANNUALIZATION_FACTOR / len(returns)) - 1

    # Risk metrics
    metrics['realized_vol'] = compute_realized_volatility(returns)
    metrics['sharpe_ratio'] = compute_sharpe_ratio(returns, risk_free_rate)
    metrics['max_drawdown'] = compute_max_drawdown(returns)

    # Volatility deviation (if target provided)
    if target_vol is not None:
        metrics['vol_deviation'] = compute_vol_deviation(metrics['realized_vol'], target_vol)
    else:
        metrics['vol_deviation'] = np.nan

    # Additional metrics
    metrics['win_rate'] = np.mean(returns > 0)
    metrics['avg_win'] = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
    metrics['avg_loss'] = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
    metrics['profit_factor'] = (
        np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))
        if np.sum(returns[returns < 0]) != 0
        else np.inf
    )

    return metrics


def run_backtest(
    returns: np.ndarray,
    predicted_vol: np.ndarray,
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
    min_scale: float = 0.0,
    risk_free_rate: float = 0.0,
    dates: Optional[pd.DatetimeIndex] = None,
    strategy_name: str = "Dynamic"
) -> Dict:
    """
    Run complete backtest for dynamic sizing.

    Args:
        returns: Daily asset returns.
        predicted_vol: Predicted volatility (annualized) for each day.
        target_vol: Target volatility (annualized).
        max_leverage: Maximum leverage cap.
        min_scale: Minimum scale (floor).
        risk_free_rate: Annualized risk-free rate.
        dates: Dates for time-series analysis.
        strategy_name: Name for the strategy.

    Returns:
        Dictionary with:
            - returns: Daily strategy returns
            - positions: Daily position sizes
            - metrics: Performance metrics
            - stats: Additional statistics
    """
    # Align inputs
    if len(returns) != len(predicted_vol):
        warnings.warn(f"Returns length ({len(returns)}) != predicted_vol length ({len(predicted_vol)})")
        # Truncate to minimum length
        min_len = min(len(returns), len(predicted_vol))
        returns = returns[:min_len]
        predicted_vol = predicted_vol[:min_len]
        if dates is not None:
            dates = dates[:min_len]

    # Run backtest
    results = backtest_dynamic_sizing(
        returns=returns,
        predicted_vol=predicted_vol,
        target_vol=target_vol,
        max_leverage=max_leverage,
        min_scale=min_scale
    )

    # Compute metrics
    metrics = compute_backtest_metrics(results['returns'], target_vol, risk_free_rate)

    # Additional stats
    stats = {
        'strategy_name': strategy_name,
        'target_vol': target_vol,
        'max_leverage': max_leverage,
        'min_scale': min_scale,
        'avg_position': np.mean(results['positions']),
        'std_position': np.std(results['positions']),
        'max_position': np.max(results['positions']),
        'min_position': np.min(results['positions']),
        'turnover': np.mean(np.abs(np.diff(results['positions']))),
    }

    return {
        'returns': results['returns'],
        'positions': results['positions'],
        'metrics': metrics,
        'stats': stats,
    }


def compare_backtests(
    returns: np.ndarray,
    predicted_vol: np.ndarray,
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
    min_scale: float = 0.0,
    risk_free_rate: float = 0.0,
    dates: Optional[pd.DatetimeIndex] = None
) -> pd.DataFrame:
    """
    Run and compare static vs dynamic backtests.

    Args:
        returns: Daily asset returns.
        predicted_vol: Predicted volatility (annualized) for each day.
        target_vol: Target volatility (annualized).
        max_leverage: Maximum leverage cap.
        min_scale: Minimum scale (floor).
        risk_free_rate: Annualized risk-free rate.
        dates: Dates for time-series analysis.

    Returns:
        DataFrame comparing static vs dynamic performance.
    """
    # Align inputs
    if len(returns) != len(predicted_vol):
        min_len = min(len(returns), len(predicted_vol))
        returns = returns[:min_len]
        predicted_vol = predicted_vol[:min_len]
        if dates is not None:
            dates = dates[:min_len]

    # Static backtest
    static_results = backtest_static_sizing(returns)
    static_metrics = compute_backtest_metrics(static_results['returns'], target_vol, risk_free_rate)

    # Dynamic backtest
    dynamic_results = backtest_dynamic_sizing(
        returns=returns,
        predicted_vol=predicted_vol,
        target_vol=target_vol,
        max_leverage=max_leverage,
        min_scale=min_scale
    )
    dynamic_metrics = compute_backtest_metrics(dynamic_results['returns'], target_vol, risk_free_rate)

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Static': static_metrics,
        'Dynamic': dynamic_metrics
    })

    # Add improvement column
    comparison['Improvement'] = comparison['Static'] - comparison['Dynamic']

    # For metrics where lower is better (vol, drawdown, deviation)
    lower_better = ['realized_vol', 'max_drawdown', 'vol_deviation']
    for metric in lower_better:
        if metric in comparison.index:
            # Positive improvement means static - dynamic > 0
            comparison.loc[metric, 'Improvement'] = comparison.loc[metric, 'Static'] - comparison.loc[metric, 'Dynamic']

    # For metrics where higher is better (sharpe, return)
    higher_better = ['sharpe_ratio', 'annualized_return', 'total_return']
    for metric in higher_better:
        if metric in comparison.index:
            # Positive improvement means dynamic - static > 0
            comparison.loc[metric, 'Improvement'] = comparison.loc[metric, 'Dynamic'] - comparison.loc[metric, 'Static']

    return comparison


def plot_cumulative_returns(
    returns_dict: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    title: str = "Cumulative Returns",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot cumulative returns for multiple strategies.

    Args:
        returns_dict: Dictionary mapping strategy_name -> returns array.
        dates: Dates for x-axis.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(dates, cumulative, label=name, linewidth=1.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_position_sizes(
    positions: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str = "Position Sizes",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot position sizes over time.

    Args:
        positions: Position sizes.
        dates: Dates for x-axis.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, positions, color='blue', linewidth=1.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Static (1x)')
    ax.axhline(y=config.TARGET_VOLATILITY / 0.20, color='green', linestyle=':', linewidth=1, alpha=0.7, label=f'Target Vol / 0.20')

    ax.set_xlabel('Date')
    ax.set_ylabel('Position Size')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_volatility_target_tracking(
    returns: np.ndarray,
    target_vol: float,
    dates: pd.DatetimeIndex,
    title: str = "Volatility Target Tracking",
    window: int = 21,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot rolling realized volatility vs target.

    Args:
        returns: Daily returns.
        target_vol: Target volatility (annualized).
        dates: Dates for x-axis.
        title: Plot title.
        window: Rolling window for realized volatility.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    # Compute rolling realized volatility
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(config.ANNUALIZATION_FACTOR)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, rolling_vol, color='blue', linewidth=1.5, label=f'Rolling Vol ({window}d)')
    ax.axhline(y=target_vol, color='red', linestyle='--', linewidth=2, label=f'Target Vol ({target_vol*100:.0f}%)')

    # Add target ± 2% bands
    ax.fill_between(
        dates,
        target_vol - 0.02,
        target_vol + 0.02,
        color='red',
        alpha=0.1,
        label='Target ± 2%'
    )

    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (Annualized)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_backtest_report(
    returns: np.ndarray,
    predicted_vol: np.ndarray,
    dates: pd.DatetimeIndex,
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
    min_scale: float = 0.0,
    risk_free_rate: float = 0.0,
    output_dir: str = 'outputs'
) -> pd.DataFrame:
    """
    Generate complete backtest report with all tables and figures.

    Args:
        returns: Daily asset returns.
        predicted_vol: Predicted volatility (annualized) for each day.
        dates: Dates for x-axis.
        target_vol: Target volatility (annualized).
        max_leverage: Maximum leverage cap.
        min_scale: Minimum scale (floor).
        risk_free_rate: Annualized risk-free rate.
        output_dir: Output directory.

    Returns:
        Comparison DataFrame.
    """
    # Create output directories
    figures_dir = os.path.join(output_dir, 'figures')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Align inputs
    if len(returns) != len(predicted_vol):
        min_len = min(len(returns), len(predicted_vol))
        returns = returns[:min_len]
        predicted_vol = predicted_vol[:min_len]
        dates = dates[:min_len]

    # Run backtests
    static_results = backtest_static_sizing(returns)
    dynamic_results = backtest_dynamic_sizing(
        returns=returns,
        predicted_vol=predicted_vol,
        target_vol=target_vol,
        max_leverage=max_leverage,
        min_scale=min_scale
    )

    # Compute metrics
    static_metrics = compute_backtest_metrics(static_results['returns'], target_vol, risk_free_rate)
    dynamic_metrics = compute_backtest_metrics(dynamic_results['returns'], target_vol, risk_free_rate)

    # Create comparison
    comparison = compare_backtests(
        returns=returns,
        predicted_vol=predicted_vol,
        target_vol=target_vol,
        max_leverage=max_leverage,
        min_scale=min_scale,
        risk_free_rate=risk_free_rate,
        dates=dates
    )

    # Save comparison
    comparison.to_csv(os.path.join(tables_dir, 'backtest_comparison.csv'))

    # Save detailed metrics
    metrics_df = pd.DataFrame({
        'Metric': list(static_metrics.keys()),
        'Static': list(static_metrics.values()),
        'Dynamic': list(dynamic_metrics.values())
    })
    metrics_df.to_csv(os.path.join(tables_dir, 'backtest_metrics.csv'), index=False)

    # Generate plots
    # Cumulative returns
    plot_cumulative_returns(
        {
            'Static': static_results['returns'],
            'Dynamic': dynamic_results['returns']
        },
        dates,
        title="Cumulative Returns: Static vs Dynamic Sizing",
        save_path=os.path.join(figures_dir, 'backtest_cumulative_returns.png')
    )

    # Position sizes
    plot_position_sizes(
        dynamic_results['positions'],
        dates,
        title="Dynamic Position Sizes Over Time",
        save_path=os.path.join(figures_dir, 'position_sizes.png')
    )

    # Volatility tracking
    plot_volatility_target_tracking(
        dynamic_results['returns'],
        target_vol,
        dates,
        title="Volatility Target Tracking (Dynamic Strategy)",
        save_path=os.path.join(figures_dir, 'volatility_tracking.png')
    )

    # Rolling volatility comparison
    # This is already covered above but we can add a combined version
    rolling_static = pd.Series(static_results['returns']).rolling(21).std() * np.sqrt(config.ANNUALIZATION_FACTOR)
    rolling_dynamic = pd.Series(dynamic_results['returns']).rolling(21).std() * np.sqrt(config.ANNUALIZATION_FACTOR)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, rolling_static, label='Static (21d rolling)', linewidth=1.5)
    ax.plot(dates, rolling_dynamic, label='Dynamic (21d rolling)', linewidth=1.5)
    ax.axhline(y=target_vol, color='red', linestyle='--', linewidth=2, label=f'Target ({target_vol*100:.0f}%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (Annualized)')
    ax.set_title('Rolling Volatility Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'rolling_volatility_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Backtest report generated in {output_dir}")
    return comparison


# Quick test function
if __name__ == "__main__":
    print("Backtest module loaded successfully.")
    print("Features: static sizing, dynamic sizing, Sharpe ratio, max drawdown, volatility tracking")
    print("Target vol:", config.TARGET_VOLATILITY)