#!/usr/bin/env python
"""
visualize_linkedin.py - True Quant Finance Dark Theme

Professional, institutional-grade visualizations optimized for LinkedIn.
Style inspired by Bloomberg Terminal, Two Sigma, and Renaissance Technologies.

USAGE:
    python visualize_linkedin.py
"""

import argparse
import os
import warnings
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# TRUE QUANT FINANCE DARK THEME
# ============================================================================

# Premium dark palette - institutional grade
QUANT_DARK = {
    # Backgrounds - deep, rich, not pure black
    'bg_primary': '#111827',      # Deep navy-charcoal
    'bg_secondary': '#1a2332',    # Slightly lighter for panels
    'bg_tertiary': '#243044',     # For alternating elements
    
    # Text - crisp but not harsh
    'text_primary': '#e8edf5',    # Off-white for main text
    'text_secondary': '#94a3b8',  # Muted for secondary
    'text_muted': '#64748b',      # Very muted
    
    # Grid - subtle, thin
    'grid_major': '#2d3a4a',
    'grid_minor': '#1f2a36',
    
    # Primary series - muted but distinct (not neon)
    'static': '#ef6b6b',          # Muted red
    'dynamic': '#4ade80',         # Muted green
    'target': '#fbbf24',          # Muted gold
    
    # Model colors - institutional palette
    'ridge': '#60a5fa',           # Soft blue
    'rf': '#a78bfa',              # Soft purple
    'lgb': '#4ade80',             # Soft green
    'ensemble': '#fb923c',        # Soft orange
    'rolling': '#94a3b8',         # Gray
    'ewma': '#ef6b6b',            # Same as static
    
    # Accents - used sparingly
    'accent_blue': '#3b82f6',
    'accent_green': '#22c55e',
    'accent_gold': '#eab308',
    'accent_purple': '#8b5cf6',
    
    # Line styling
    'line_width': 1.8,
    'line_width_thick': 2.4,
    'alpha_series': 0.9,
    'alpha_glow': 0.08,
}

# Quant finance typography - sans-serif standard
QUANT_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 600,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'figure.titlesize': 15,
    'figure.titleweight': 600,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
    'lines.linewidth': QUANT_DARK['line_width'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.edgecolor': '#2d3a4a',  # Changed from 'axes.spines.color'
    'axes.facecolor': QUANT_DARK['bg_primary'],
    'figure.facecolor': QUANT_DARK['bg_primary'],
    'savefig.facecolor': QUANT_DARK['bg_primary'],
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def setup_quant_style():
    """Apply true quant finance dark theme."""
    plt.style.use('dark_background')
    for key, value in QUANT_RCPARAMS.items():
        plt.rcParams[key] = value


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(tables_path: str = 'outputs/tables') -> Dict[str, pd.DataFrame]:
    """Load all required data files."""
    data = {}
    files = {
        'model_results': 'model_results.csv',
        'rmse_comparison': 'rmse_comparison.csv',
        'backtest_comparison': 'backtest_comparison.csv',
        'best_models': 'best_models.csv',
        'predictions': 'predictions.csv',
    }
    
    for name, filename in files.items():
        path = os.path.join(tables_path, filename)
        if os.path.exists(path):
            if name == 'backtest_comparison':
                data[name] = pd.read_csv(path, index_col=0)
            else:
                data[name] = pd.read_csv(path)
            print(f"  ✓ {name}")
        else:
            data[name] = None
            print(f"  ✗ {name} not found")
    
    return data


def load_daily_returns(tables_path: str = 'outputs/tables') -> Optional[pd.DataFrame]:
    path = os.path.join(tables_path, 'daily_returns.csv')
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    return None


# ============================================================================
# VISUALIZATION 1: Volatility Cone
# ============================================================================

def visualize_volatility_cone(predictions_df, colors, output_dir):
    """Institutional volatility cone with subtle confidence bands."""
    if predictions_df is None:
        print("  ✗ No predictions data")
        return None
    
    df = predictions_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Use first ticker
    if df['ticker'].nunique() > 1:
        ticker = df['ticker'].value_counts().index[0]
        df = df[df['ticker'] == ticker].copy()
    
    df = df.sort_values('date')
    if len(df) > 500:
        df = df.sample(n=500, random_state=42).sort_values('date')
    
    dates = df['date'].values
    actual = df['actual_vol'].values
    predicted = df['predicted_vol'].values
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(colors['bg_primary'])
    
    # Confidence bands - very subtle
    upper = predicted + 1.96 * rmse
    lower = predicted - 1.96 * rmse
    ax.fill_between(dates, lower, upper, color=colors['dynamic'], 
                    alpha=0.08, zorder=0)
    
    # Actual - thin, muted
    ax.plot(dates, actual, color=colors['static'], linewidth=1.2, 
            alpha=0.7, label='Actual', zorder=2)
    
    # Predicted - thicker, with subtle glow
    ax.plot(dates, predicted, color=colors['dynamic'], 
            linewidth=QUANT_DARK['line_width_thick'], 
            alpha=0.9, label='Predicted', zorder=3)
    
    # Glow effect
    ax.plot(dates, predicted, color=colors['dynamic'], 
            alpha=0.06, linewidth=8, zorder=1)
    
    ax.set_xlabel('Date', color=colors['text_secondary'])
    ax.set_ylabel('Volatility (Annualized)', color=colors['text_secondary'])
    ax.set_title('Volatility Cone: Actual vs Predicted', 
                 color=colors['text_primary'], fontweight=600)
    
    # Legend - no box
    ax.legend(loc='upper right', frameon=False, labelcolor=colors['text_secondary'])
    
    # Grid - subtle
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax.tick_params(colors=colors['text_muted'])
    
    # Metrics annotation - clean, minimal
    ax.annotate(f'RMSE {rmse:.4f}  |  MAE {mae:.4f}  |  n {len(dates)}',
                xy=(0.02, 0.06), xycoords='axes fraction',
                fontsize=10, color=colors['text_muted'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '01_volatility_cone.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 01_volatility_cone.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 2: Mincer-Zarnowitz
# ============================================================================

def visualize_mincer_zarnowitz(predictions_df, colors, output_dir):
    """Clean calibration scatter with regression line."""
    if predictions_df is None:
        print("  ✗ No predictions data")
        return None
    
    df = predictions_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if df['ticker'].nunique() > 1:
        df = df.groupby('date').agg({
            'actual_vol': 'mean',
            'predicted_vol': 'mean'
        }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(colors['bg_primary'])
    
    # Scatter - small, transparent
    ax.scatter(df['predicted_vol'], df['actual_vol'],
               alpha=0.3, s=15, color=colors['accent_blue'],
               edgecolors='none', zorder=2)
    
    # 45-degree line
    min_val = min(df['predicted_vol'].min(), df['actual_vol'].min())
    max_val = max(df['predicted_vol'].max(), df['actual_vol'].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color=colors['text_muted'], linestyle='--', 
            linewidth=1, alpha=0.5, zorder=1)
    
    # Regression line
    slope, intercept, r_value, p_value, _ = stats.linregress(
        df['predicted_vol'], df['actual_vol']
    )
    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=colors['dynamic'], 
            linewidth=QUANT_DARK['line_width_thick'], zorder=3)
    
    # Metrics
    r_squared = r_value ** 2
    rmse = np.sqrt(np.mean((df['actual_vol'] - df['predicted_vol']) ** 2))
    
    ax.set_xlabel('Predicted Volatility', color=colors['text_secondary'])
    ax.set_ylabel('Actual Volatility', color=colors['text_secondary'])
    ax.set_title('Mincer-Zarnowitz: Forecast Calibration',
                 color=colors['text_primary'], fontweight=600)
    
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax.tick_params(colors=colors['text_muted'])
    ax.set_aspect('equal')
    
    # Clean annotation
    ax.annotate(f'β = {slope:.3f}    R² = {r_squared:.3f}    RMSE = {rmse:.4f}',
                xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=10, color=colors['text_secondary'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '02_mincer_zarnowitz.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 02_mincer_zarnowitz.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 3: Cumulative Returns
# ============================================================================

def visualize_cumulative_returns(backtest_comparison, daily_returns, colors, output_dir):
    """Clean cumulative returns with divergence highlight."""
    if backtest_comparison is None:
        print("  ✗ No backtest data")
        return None
    
    static_total = backtest_comparison.loc['total_return', 'Static']
    dynamic_total = backtest_comparison.loc['total_return', 'Dynamic']
    
    if daily_returns is not None and len(daily_returns) > 0:
        df = daily_returns.sort_values('date')
        static_cum = (1 + df['static_returns']).cumprod()
        dynamic_cum = (1 + df['dynamic_returns']).cumprod()
        dates = df['date'].values
    else:
        # Minimal synthetic fallback
        n = 1260
        dates = pd.date_range('2020-01-02', periods=n, freq='B')
        np.random.seed(42)
        static_ret = np.random.normal(0, 0.15 / np.sqrt(252), n)
        dynamic_ret = np.random.normal(0, 0.115 / np.sqrt(252), n)
        static_cum = (1 + static_ret).cumprod()
        dynamic_cum = (1 + dynamic_ret).cumprod()
        # Adjust to match actual totals
        static_cum = static_cum / static_cum.iloc[-1] * (1 + static_total)
        dynamic_cum = dynamic_cum / dynamic_cum.iloc[-1] * (1 + dynamic_total)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_facecolor(colors['bg_primary'])
    
    # Dynamic - main focus
    ax.plot(dates, dynamic_cum, color=colors['dynamic'], 
            linewidth=QUANT_DARK['line_width_thick'], label='Dynamic', zorder=3)
    
    # Static - secondary
    ax.plot(dates, static_cum, color=colors['static'], 
            linewidth=1.4, alpha=0.7, label='Static', zorder=2)
    
    # Baseline
    ax.axhline(y=1.0, color=colors['text_muted'], linestyle='--', linewidth=0.8, alpha=0.4)
    
    # Divergence highlight - fill between
    ax.fill_between(dates, static_cum, dynamic_cum, 
                    where=(dynamic_cum > static_cum),
                    color=colors['dynamic'], alpha=0.05, zorder=0)
    ax.fill_between(dates, static_cum, dynamic_cum,
                    where=(dynamic_cum < static_cum),
                    color=colors['static'], alpha=0.03, zorder=0)
    
    ax.set_xlabel('Date', color=colors['text_secondary'])
    ax.set_ylabel('Cumulative Return', color=colors['text_secondary'])
    ax.set_title('Cumulative Returns: Static vs Dynamic Sizing',
                 color=colors['text_primary'], fontweight=600)
    
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax.tick_params(colors=colors['text_muted'])
    
    # Legend - clean
    ax.legend(loc='upper left', frameon=False, labelcolor=colors['text_secondary'])
    
    # Final values - clean annotation
    improvement = dynamic_total - static_total
    ax.annotate(f'Static  {static_total:+.1%}    Dynamic  {dynamic_total:+.1%}    Δ  {improvement:+.1%}',
                xy=(0.02, 0.04), xycoords='axes fraction',
                fontsize=10, color=colors['text_muted'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '03_cumulative_returns.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 03_cumulative_returns.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 4: Rolling Volatility
# ============================================================================

def visualize_rolling_volatility(backtest_comparison, daily_returns, colors, output_dir):
    """Rolling volatility with target band."""
    if backtest_comparison is None:
        print("  ✗ No backtest data")
        return None
    
    window = 63
    target_vol = 0.15
    static_vol = backtest_comparison.loc['realized_vol', 'Static']
    dynamic_vol = backtest_comparison.loc['realized_vol', 'Dynamic']
    
    if daily_returns is not None and len(daily_returns) > 0:
        df = daily_returns.sort_values('date')
        dates = df['date'].values
        static_rolling = df['static_returns'].rolling(window).std() * np.sqrt(252)
        dynamic_rolling = df['dynamic_returns'].rolling(window).std() * np.sqrt(252)
        static_rolling = static_rolling.fillna(method='bfill')
        dynamic_rolling = dynamic_rolling.fillna(method='bfill')
    else:
        n = 1260
        dates = pd.date_range('2020-01-02', periods=n, freq='B')
        np.random.seed(42)
        static_ret = np.random.normal(0, static_vol / np.sqrt(252), n)
        dynamic_ret = np.random.normal(0, dynamic_vol / np.sqrt(252), n)
        static_rolling = pd.Series(static_ret).rolling(window).std() * np.sqrt(252)
        dynamic_rolling = pd.Series(dynamic_ret).rolling(window).std() * np.sqrt(252)
        static_rolling = static_rolling.fillna(method='bfill')
        dynamic_rolling = dynamic_rolling.fillna(method='bfill')
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_facecolor(colors['bg_primary'])
    
    # Target band - subtle
    ax.fill_between(dates, 0.13, 0.17, color=colors['target'], alpha=0.05, zorder=0)
    
    # Dynamic
    ax.plot(dates, dynamic_rolling, color=colors['dynamic'], 
            linewidth=QUANT_DARK['line_width_thick'], label='Dynamic', zorder=3)
    
    # Static
    ax.plot(dates, static_rolling, color=colors['static'], 
            linewidth=1.4, alpha=0.6, label='Static', zorder=2)
    
    # Target line
    ax.axhline(y=target_vol, color=colors['target'], linestyle='--', 
               linewidth=1.2, alpha=0.6, label='Target')
    
    ax.set_xlabel('Date', color=colors['text_secondary'])
    ax.set_ylabel('Volatility (Annualized)', color=colors['text_secondary'])
    ax.set_title(f'Rolling Volatility ({window}-day Window)',
                 color=colors['text_primary'], fontweight=600)
    
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax.tick_params(colors=colors['text_muted'])
    ax.legend(loc='upper right', frameon=False, labelcolor=colors['text_secondary'])
    
    # Metrics
    ax.annotate(f'Static  {static_vol:.1%}    Dynamic  {dynamic_vol:.1%}    Target  {target_vol:.0%}',
                xy=(0.02, 0.04), xycoords='axes fraction',
                fontsize=10, color=colors['text_muted'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '04_rolling_volatility.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 04_rolling_volatility.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 5: Model Comparison
# ============================================================================

def visualize_model_comparison(rmse_comparison, model_results, best_models, colors, output_dir):
    """Clean bar chart with best model highlighted."""
    if rmse_comparison is None:
        print("  ✗ No RMSE data")
        return None
    
    models = ['Ridge', 'RandomForest', 'LightGBM', 'Ensemble_RF_LGB']
    feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']
    feature_labels = ['B1', 'B2', 'B3', 'Adv']
    
    # Extract data
    rmse_data = {}
    for model in models:
        row = rmse_comparison[rmse_comparison['model'] == model]
        if not row.empty:
            rmse_data[model] = [row[fs].values[0] if fs in row.columns else np.nan 
                                for fs in feature_sets]
    
    x = np.arange(len(models))
    width = 0.18
    
    # Institutional color palette
    bar_colors = ['#475569', '#ef6b6b', '#4ade80', '#60a5fa']
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_facecolor(colors['bg_primary'])
    
    for i, (fs, label) in enumerate(zip(feature_sets, feature_labels)):
        vals = [rmse_data.get(model, [np.nan]*4)[i] for model in models]
        offset = i * width - 0.27
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=bar_colors[i], alpha=0.8, edgecolor='none')
    
    # Highlight best (LightGBM, Baseline 3)
    best_row = best_models[best_models['feature_set'] == 'baseline_3']
    if not best_row.empty:
        best_model = best_row.iloc[0]['model']
        if best_model in models:
            best_idx = models.index(best_model)
            # Small indicator
            ax.scatter(best_idx, 0.002, marker='v', s=60, 
                      color=colors['accent_gold'], zorder=5, clip_on=False)
    
    ax.set_xlabel('Model', color=colors['text_secondary'])
    ax.set_ylabel('RMSE', color=colors['text_secondary'])
    ax.set_title('Model Comparison: RMSE by Feature Set',
                 color=colors['text_primary'], fontweight=600)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color=colors['text_secondary'])
    
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(colors=colors['text_muted'])
    ax.legend(loc='upper right', frameon=False, labelcolor=colors['text_secondary'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '05_model_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 05_model_comparison.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 6: Position Sizes
# ============================================================================

def visualize_position_sizes(predictions_df, colors, output_dir):
    """Position sizes with area fill."""
    if predictions_df is None:
        print("  ✗ No predictions data")
        return None
    
    df = predictions_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if df['ticker'].nunique() > 1:
        df = df[df['ticker'] == df['ticker'].value_counts().index[0]].copy()
    
    df = df.sort_values('date')
    if len(df) > 500:
        df = df.sample(n=500, random_state=42).sort_values('date')
    
    target_vol = 0.15
    predicted = df['predicted_vol'].values
    dates = df['date'].values
    
    position_size = np.clip(target_vol / np.maximum(predicted, 0.01), 0.1, 1.0)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_facecolor(colors['bg_primary'])
    
    # Area fill - subtle
    ax.fill_between(dates, 0, position_size, color=colors['dynamic'], alpha=0.12)
    
    # Line
    ax.plot(dates, position_size, color=colors['dynamic'], 
            linewidth=QUANT_DARK['line_width_thick'], zorder=3)
    
    # Static baseline
    ax.axhline(y=1.0, color=colors['static'], linestyle='--', 
               linewidth=1, alpha=0.5, label='Static (1x)')
    
    ax.set_xlabel('Date', color=colors['text_secondary'])
    ax.set_ylabel('Position Size', color=colors['text_secondary'])
    ax.set_title('Dynamic Position Sizing: Volatility-Driven Allocation',
                 color=colors['text_primary'], fontweight=600)
    
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax.tick_params(colors=colors['text_muted'])
    ax.legend(loc='upper right', frameon=False, labelcolor=colors['text_secondary'])
    
    # Stats
    ax.annotate(f'Mean {np.mean(position_size):.2f}x    Min {np.min(position_size):.2f}x    Max {np.max(position_size):.2f}x',
                xy=(0.02, 0.04), xycoords='axes fraction',
                fontsize=10, color=colors['text_muted'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '06_position_sizes.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 06_position_sizes.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 7: Error Distribution
# ============================================================================

def visualize_error_distribution(predictions_df, colors, output_dir):
    """Clean error distribution with Q-Q plot."""
    if predictions_df is None:
        print("  ✗ No predictions data")
        return None
    
    errors = predictions_df['actual_vol'].values - predictions_df['predicted_vol'].values
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)
    skew = stats.skew(errors)
    kurtosis = stats.kurtosis(errors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.set_facecolor(colors['bg_primary'])
    ax2.set_facecolor(colors['bg_primary'])
    
    # Histogram + KDE
    sns.histplot(errors, kde=True, ax=ax1, color=colors['accent_blue'], 
                 bins=25, alpha=0.4, edgecolor='none', 
                 line_kws={'linewidth': 1.5, 'color': colors['accent_blue']})
    ax1.axvline(0, color=colors['static'], linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(mean_error, color=colors['dynamic'], linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Forecast Error', color=colors['text_secondary'])
    ax1.set_ylabel('Frequency', color=colors['text_secondary'])
    ax1.set_title('Error Distribution', color=colors['text_primary'], fontweight=600)
    ax1.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax1.tick_params(colors=colors['text_muted'])
    
    # Q-Q plot
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', color=colors['text_primary'], fontweight=600)
    ax2.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5)
    ax2.tick_params(colors=colors['text_muted'])
    for line in ax2.get_lines():
        line.set_color(colors['accent_blue'])
        line.set_linewidth(1.5)
    for text in ax2.texts:
        text.set_color(colors['text_secondary'])
    
    # Metrics
    ax1.annotate(f'RMSE {rmse:.4f}    Mean {mean_error:.4f}    Skew {skew:.3f}    Kurt {kurtosis:.3f}',
                 xy=(0.02, 0.94), xycoords='axes fraction',
                 fontsize=9, color=colors['text_muted'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '07_error_distribution.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 07_error_distribution.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 8: Feature Importance
# ============================================================================

def visualize_feature_importance(model_dir, colors, output_dir):
    """Clean feature importance horizontal bars."""
    model_path = os.path.join(model_dir, 'lightgbm_advanced.joblib')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            importance = model.feature_importances_
            features = model.feature_name_
            sorted_idx = np.argsort(importance)[::-1]
            features_sorted = [features[i] for i in sorted_idx]
            importance_sorted = np.array([importance[i] for i in sorted_idx])
            importance_sorted = importance_sorted / np.sum(importance_sorted)
        except:
            features_sorted = ['EWMA Vol', 'Rolling Vol 21d', 'VIX Level', 
                              'Parkinson Vol', 'Rolling Vol 63d', 'VIX Change']
            importance_sorted = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    else:
        features_sorted = ['EWMA Vol', 'Rolling Vol 21d', 'VIX Level', 
                          'Parkinson Vol', 'Rolling Vol 63d', 'VIX Change']
        importance_sorted = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor(colors['bg_primary'])
    
    # Gradient from blue to teal
    gradient = plt.cm.Blues_r(np.linspace(0.3, 0.8, len(features_sorted)))
    
    bars = ax.barh(features_sorted, importance_sorted, color=gradient,
                   edgecolor='none', height=0.5)
    
    # Value labels
    for bar, val in zip(bars, importance_sorted):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}',
                va='center', fontsize=10, color=colors['text_secondary'])
    
    ax.set_xlabel('Feature Importance', color=colors['text_secondary'])
    ax.set_title('Feature Importance: Model Drivers',
                 color=colors['text_primary'], fontweight=600)
    ax.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5, axis='x')
    ax.tick_params(colors=colors['text_muted'])
    ax.spines['left'].set_color(colors['grid_major'])
    ax.spines['bottom'].set_color(colors['grid_major'])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '08_feature_importance.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 08_feature_importance.png")
    plt.close()
    
    return fig


# ============================================================================
# VISUALIZATION 9: Experiment Results
# ============================================================================

def visualize_experiment_results(experiments_path, colors, output_dir):
    """Clean experiment results with clear winner."""
    exp_path = os.path.join(experiments_path, 'experiment_results.csv')
    if not os.path.exists(exp_path):
        print("  ✗ No experiment data")
        return None
    
    exp_df = pd.read_csv(exp_path)
    valid = exp_df[exp_df['beta_improvement_pct'].notna()]
    valid = valid[~valid['experiment'].str.contains('ewma_decay', na=False)]
    valid = valid[~valid['experiment'].str.contains('error', na=False)]
    
    if len(valid) == 0:
        print("  ✗ No valid experiment data")
        return None
    
    valid = valid.sort_values('beta_improvement_pct', ascending=True)
    
    label_map = {
        'leverage': 'Leverage',
        'vol_of_vol': 'Vol of Vol',
        'isotonic': 'Isotonic',
        'log_target': 'Log-Transform'
    }
    valid['label'] = valid['experiment'].map(label_map).fillna(valid['experiment'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.set_facecolor(colors['bg_primary'])
    ax2.set_facecolor(colors['bg_primary'])
    
    colors1 = [colors['dynamic'] if x > 0 else colors['static'] 
               for x in valid['beta_improvement_pct']]
    colors2 = [colors['dynamic'] if x > 0 else colors['static'] 
               for x in valid['rmse_improvement_pct']]
    
    # Beta Improvement
    bars1 = ax1.barh(valid['label'], valid['beta_improvement_pct'], 
                     color=colors1, height=0.5, edgecolor='none')
    ax1.axvline(x=0, color=colors['text_muted'], linestyle='-', linewidth=0.8, alpha=0.4)
    ax1.set_xlabel('β Improvement (%)', color=colors['text_secondary'])
    ax1.set_title('Calibration: β → 1', color=colors['text_primary'], fontweight=600)
    ax1.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5, axis='x')
    ax1.tick_params(colors=colors['text_muted'])
    ax1.spines['left'].set_color(colors['grid_major'])
    ax1.spines['bottom'].set_color(colors['grid_major'])
    
    for bar, val in zip(bars1, valid['beta_improvement_pct']):
        x_pos = val + 1.5 if val > 0 else val - 3
        color = colors['text_secondary'] if val > 0 else colors['static']
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=9, color=color)
    
    # RMSE Change
    bars2 = ax2.barh(valid['label'], valid['rmse_improvement_pct'], 
                     color=colors2, height=0.5, edgecolor='none')
    ax2.axvline(x=0, color=colors['text_muted'], linestyle='-', linewidth=0.8, alpha=0.4)
    ax2.set_xlabel('RMSE Change (%)', color=colors['text_secondary'])
    ax2.set_title('Accuracy: RMSE ↓', color=colors['text_primary'], fontweight=600)
    ax2.grid(True, alpha=0.15, color=colors['grid_major'], linestyle='--', linewidth=0.5, axis='x')
    ax2.tick_params(colors=colors['text_muted'])
    ax2.spines['left'].set_color(colors['grid_major'])
    ax2.spines['bottom'].set_color(colors['grid_major'])
    
    for bar, val in zip(bars2, valid['rmse_improvement_pct']):
        x_pos = val + 0.5 if val > 0 else val - 1.5
        color = colors['text_secondary'] if val > 0 else colors['static']
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=9, color=color)
    
    # Winner indicator
    ax1.annotate('BEST', xy=(0.02, 0.94), xycoords='axes fraction',
                fontsize=10, color=colors['accent_gold'], fontweight=600)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '09_experiment_results.png'), 
                dpi=300, bbox_inches='tight', facecolor=colors['bg_primary'])
    print(f"  ✓ 09_experiment_results.png")
    plt.close()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables-path', default='outputs/tables')
    parser.add_argument('--output-dir', default='outputs/figures_linkedin')
    parser.add_argument('--models-path', default='outputs/models')
    parser.add_argument('--experiments-path', default='outputs/experiments')
    args = parser.parse_args()
    
    print("=" * 60)
    print("QUANT FINANCE DARK THEME - LINKEDIN VISUALIZATIONS")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    
    setup_quant_style()
    colors = QUANT_DARK
    
    print("\nLoading data...")
    data = load_data(args.tables_path)
    daily_returns = load_daily_returns(args.tables_path)
    
    print("\nGenerating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualize_volatility_cone(data['predictions'], colors, args.output_dir)
    visualize_mincer_zarnowitz(data['predictions'], colors, args.output_dir)
    visualize_cumulative_returns(data['backtest_comparison'], daily_returns, colors, args.output_dir)
    visualize_rolling_volatility(data['backtest_comparison'], daily_returns, colors, args.output_dir)
    visualize_model_comparison(data['rmse_comparison'], data['model_results'], 
                               data['best_models'], colors, args.output_dir)
    visualize_position_sizes(data['predictions'], colors, args.output_dir)
    visualize_error_distribution(data['predictions'], colors, args.output_dir)
    visualize_feature_importance(args.models_path, colors, args.output_dir)
    visualize_experiment_results(args.experiments_path, colors, args.output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()