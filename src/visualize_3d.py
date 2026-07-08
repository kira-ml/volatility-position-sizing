#!/usr/bin/env python
"""
3D Visualizations for Volatility Forecasting Project.

Creates eye-catching 3D plots for LinkedIn/Instagram using REAL data:
1. 3D Surface: RMSE by Model and Feature Set
2. 3D Scatter: Actual vs Predicted Volatility
3. 3D Bar: Dynamic vs Static Performance Comparison

Usage:
    python src/visualize_3d.py
"""

import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_dark_style() -> Dict:
    """Set dark quant finance style for 3D visualizations."""
    plt.style.use('dark_background')
    
    colors = {
        'background': '#0a0a1a',
        'text': '#e0e0e0',
        'grid': '#2a2a4a',
        'surface_cmap': 'plasma',
        'scatter_cmap': 'cool',
        'bar_cmap': ['#00d4ff', '#ff6b6b'],
        'highlight': '#f1c40f',
        'surface': '#1abc9c'
    }
    
    plt.rcParams['figure.facecolor'] = colors['background']
    plt.rcParams['axes.facecolor'] = colors['background']
    plt.rcParams['text.color'] = colors['text']
    plt.rcParams['axes.labelcolor'] = colors['text']
    plt.rcParams['xtick.color'] = colors['text']
    plt.rcParams['ytick.color'] = colors['text']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    
    return colors


def load_data(tables_path: str = 'outputs/tables') -> Dict:
    """Load required data files."""
    data = {}
    
    files = {
        'model_results': os.path.join(tables_path, 'model_results.csv'),
        'predictions': os.path.join(tables_path, 'predictions.csv'),
        'backtest': os.path.join(tables_path, 'backtest_comparison.csv'),
        'rmse_comparison': os.path.join(tables_path, 'rmse_comparison.csv'),
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            print(f"Loaded: {name} ({len(data[name])} rows)")
        else:
            print(f"Warning: {path} not found")
            data[name] = None
    
    return data


def visualize_3d_rmse_surface(
    model_results: pd.DataFrame,
    colors: Dict,
    output_dir: str = 'outputs/figures'
) -> plt.Figure:
    """
    3D Surface: RMSE by Model and Feature Set.
    
    Data Source: model_results.csv (REAL DATA)
    X: Model (Ridge, RandomForest, LightGBM, Ensemble)
    Y: Feature Set (baseline_1, baseline_2, baseline_3, advanced)
    Z: RMSE
    """
    # Filter to relevant models
    models = ['Ridge', 'RandomForest', 'LightGBM', 'Ensemble_RF_LGB']
    feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']
    
    # Extract RMSE values from REAL data
    rmse_matrix = []
    for model in models:
        row = model_results[model_results['model'] == model]
        if not row.empty:
            vals = []
            for fs in feature_sets:
                val = row[row['feature_set'] == fs]['rmse'].values
                vals.append(val[0] if len(val) > 0 else np.nan)
            rmse_matrix.append(vals)
        else:
            rmse_matrix.append([np.nan] * len(feature_sets))
    
    rmse_matrix = np.array(rmse_matrix)
    
    # Create mesh grid
    x = np.arange(len(models))
    y = np.arange(len(feature_sets))
    X, Y = np.meshgrid(x, y)
    Z = rmse_matrix.T
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.plasma,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        rstride=1,
        cstride=1
    )
    
    # Find best (lowest RMSE) and mark it
    best_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.scatter(
        best_idx[1], best_idx[0], Z[best_idx],
        color=colors['highlight'],
        s=200,
        edgecolor='white',
        linewidth=2,
        label='Best Model'
    )
    
    # Labels
    ax.set_xlabel('Model', fontweight='bold', labelpad=15)
    ax.set_ylabel('Feature Set', fontweight='bold', labelpad=15)
    ax.set_zlabel('RMSE', fontweight='bold', labelpad=15)
    ax.set_title('3D RMSE Surface: Model Performance by Feature Set', 
                 fontweight='bold', fontsize=18, pad=20)
    
    # Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(['Ridge', 'RF', 'LGB', 'Ensemble'], rotation=30, ha='right')
    ax.set_yticks(y)
    ax.set_yticklabels(['Baseline 1', 'Baseline 2', 'Baseline 3', 'Advanced'], rotation=20)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('RMSE', fontweight='bold')
    
    # Annotate best value
    best_val = Z[best_idx]
    ax.text(
        best_idx[1], best_idx[0], best_val + 0.01,
        f'Best: {best_val:.4f}',
        color='white',
        fontweight='bold',
        ha='center',
        va='bottom',
        fontsize=12
    )
    
    ax.view_init(elev=25, azim=-45)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '3d_rmse_surface_dark.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a1a')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def visualize_3d_prediction_scatter(
    predictions_df: pd.DataFrame,
    colors: Dict,
    output_dir: str = 'outputs/figures'
) -> plt.Figure:
    """
    3D Scatter: Actual vs Predicted Volatility with Error.
    
    Data Source: predictions.csv (REAL DATA)
    X: Predicted Volatility
    Y: Actual Volatility
    Z: Date (or Error Magnitude)
    Color: Error Magnitude (|Actual - Predicted|)
    """
    if predictions_df is None or len(predictions_df) == 0:
        print("Warning: No predictions data. Skipping 3D scatter.")
        return None
    
    # Convert date to datetime
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    # Use first ticker's data
    tickers = predictions_df['ticker'].unique()
    df = predictions_df[predictions_df['ticker'] == tickers[0]].copy()
    df = df.sort_values('date')
    
    # Calculate error
    df['error'] = np.abs(df['actual_vol'] - df['predicted_vol'])
    
    # Convert date to numeric (days since first date)
    df['date_num'] = (df['date'] - df['date'].min()).dt.days.values
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(
        df['predicted_vol'].values,
        df['actual_vol'].values,
        df['date_num'].values,
        c=df['error'].values,
        cmap=cm.coolwarm,
        s=30,
        alpha=0.8,
        edgecolors='none'
    )
    
    # Perfect prediction line
    min_val = min(df['predicted_vol'].min(), df['actual_vol'].min())
    max_val = max(df['predicted_vol'].max(), df['actual_vol'].max())
    x_line = np.linspace(min_val, max_val, 50)
    y_line = x_line
    z_line = np.ones_like(x_line) * df['date_num'].median()
    ax.plot(x_line, y_line, z_line, color='#f1c40f', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # Labels
    ax.set_xlabel('Predicted Volatility', fontweight='bold', labelpad=15)
    ax.set_ylabel('Actual Volatility', fontweight='bold', labelpad=15)
    ax.set_zlabel('Time (Days from Start)', fontweight='bold', labelpad=15)
    ax.set_title('3D Forecast Accuracy: Actual vs Predicted Volatility', 
                 fontweight='bold', fontsize=18, pad=20)
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Forecast Error', fontweight='bold')
    
    ax.view_init(elev=20, azim=-30)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '3d_prediction_scatter_dark.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a1a')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def visualize_3d_performance_bars(
    backtest_data: pd.DataFrame,
    colors: Dict,
    output_dir: str = 'outputs/figures'
) -> plt.Figure:
    """
    3D Bar Chart: Dynamic vs Static Performance.
    
    Data Source: backtest_comparison.csv (REAL DATA)
    X: Metrics (Return, Sharpe, Drawdown, Vol)
    Y: Strategy (Static, Dynamic)
    Z: Value
    """
    # Ensure backtest data has the right index
    if 'Unnamed: 0' in backtest_data.columns:
        backtest_data = backtest_data.set_index('Unnamed: 0')
    
    # Extract metrics
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'realized_vol']
    labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Realized Vol']
    
    static_vals = []
    dynamic_vals = []
    
    for metric in metrics:
        if metric in backtest_data.index:
            static_vals.append(float(backtest_data.loc[metric, 'Static']))
            dynamic_vals.append(float(backtest_data.loc[metric, 'Dynamic']))
    
    if len(static_vals) == 0:
        print("Warning: No metrics found in backtest data. Skipping 3D bars.")
        return None
    
    # Bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dynamic bars (front)
    ax.bar3d(
        x - width/2, -0.1, 0,
        width, 0.6, dynamic_vals,
        color='#00d4ff',
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Static bars (back)
    ax.bar3d(
        x + width/2, 0.1, 0,
        width, 0.6, static_vals,
        color='#ff6b6b',
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Labels
    ax.set_xlabel('Metric', fontweight='bold', labelpad=15)
    ax.set_ylabel('Strategy', fontweight='bold', labelpad=15)
    ax.set_zlabel('Value', fontweight='bold', labelpad=15)
    ax.set_title('3D Performance Comparison: Dynamic vs Static Sizing', 
                 fontweight='bold', fontsize=18, pad=20)
    
    # Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks([-0.1, 0.1])
    ax.set_yticklabels(['Dynamic', 'Static'])
    
    ax.view_init(elev=25, azim=45)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '3d_performance_bars_dark.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a1a')
    print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def main():
    """Generate all 3D visualizations using REAL data."""
    print("=" * 60)
    print("3D VISUALIZATIONS (DARK MODE - QUANT FINANCE)")
    print("=" * 60)
    
    # Set style
    colors = set_dark_style()
    
    # Load data
    data = load_data('outputs/tables')
    
    # Output directory
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating 3D visualizations...")
    print("-" * 40)
    
    # 1. 3D RMSE Surface
    if data['model_results'] is not None and len(data['model_results']) > 0:
        visualize_3d_rmse_surface(data['model_results'], colors, output_dir)
    else:
        print("Warning: model_results.csv empty. Skipping 3D surface.")
    
    # 2. 3D Prediction Scatter
    if data['predictions'] is not None and len(data['predictions']) > 0:
        visualize_3d_prediction_scatter(data['predictions'], colors, output_dir)
    else:
        print("Warning: predictions.csv empty. Skipping 3D scatter.")
    
    # 3. 3D Performance Bars
    if data['backtest'] is not None and len(data['backtest']) > 0:
        visualize_3d_performance_bars(data['backtest'], colors, output_dir)
    else:
        print("Warning: backtest_comparison.csv empty. Skipping 3D bars.")
    
    print("\n" + "=" * 60)
    print("3D VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - 3d_rmse_surface_dark.png")
    print("  - 3d_prediction_scatter_dark.png")
    print("  - 3d_performance_bars_dark.png")
    print("=" * 60)


if __name__ == "__main__":
    main()