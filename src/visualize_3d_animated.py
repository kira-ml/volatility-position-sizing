#!/usr/bin/env python
"""
Animated 3D Visualization for LinkedIn/Instagram.

Creates a rotating 3D surface plot exported as GIF.
Perfect for social media engagement hooks.

Usage:
    python src/visualize_3d_animated.py
"""

import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


def load_data(tables_path: str = 'outputs/tables') -> Dict:
    """Load required data files."""
    import pandas as pd
    data = {}
    
    files = {
        'model_results': os.path.join(tables_path, 'model_results.csv'),
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            print(f"Loaded: {name} ({len(data[name])} rows)")
        else:
            print(f"Warning: {path} not found")
            data[name] = None
    
    return data


def create_animated_3d_surface(
    model_results: pd.DataFrame,
    output_dir: str = 'outputs/figures',
    duration: int = 8,
    fps: int = 30
) -> str:
    """
    Create animated 3D surface plot (RMSE by Model × Feature Set).
    
    Args:
        model_results: DataFrame with model results
        output_dir: Output directory
        duration: Video duration in seconds
        fps: Frames per second
    
    Returns:
        Path to GIF file
    """
    from matplotlib.animation import FuncAnimation
    
    # Prepare data
    models = ['Ridge', 'RandomForest', 'LightGBM', 'Ensemble_RF_LGB']
    feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']
    
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
    
    x = np.arange(len(models))
    y = np.arange(len(feature_sets))
    X, Y = np.meshgrid(x, y)
    Z = rmse_matrix.T
    
    # Color map for bars
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    # Dark style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    
    # Create initial bar plot
    colors_flat = Z.flatten()
    norm = Normalize(vmin=colors_flat.min(), vmax=colors_flat.max())
    cmap = cm.plasma
    
    # Bar plot
    bar_width = 0.6
    bar_depth = 0.6
    
    bars = []
    for i in range(len(x)):
        for j in range(len(y)):
            val = Z[j, i]
            if not np.isnan(val):
                color = cmap(norm(val))
                bar = ax.bar3d(
                    i - bar_width/2, j - bar_depth/2, 0,
                    bar_width, bar_depth, val,
                    color=color,
                    alpha=0.85,
                    edgecolor='none'
                )
                bars.append(bar)
    
    # Labels
    ax.set_xlabel('Model', fontweight='bold', labelpad=15, color='white')
    ax.set_ylabel('Feature Set', fontweight='bold', labelpad=15, color='white')
    ax.set_zlabel('RMSE', fontweight='bold', labelpad=15, color='white')
    ax.set_title('🎯 Model Performance: RMSE by Model & Feature Set', 
                 fontweight='bold', fontsize=20, pad=25, color='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Ridge', 'RF', 'LGB', 'Ensemble'], rotation=30, ha='right', color='white')
    ax.set_yticks(y)
    ax.set_yticklabels(['Baseline 1', 'Baseline 2', 'Baseline 3', 'Advanced'], 
                       rotation=20, color='white')
    ax.tick_params(colors='white')
    
    # Grid
    ax.grid(True, alpha=0.2)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('RMSE', fontweight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Find best model
    best_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    best_val = Z[best_idx]
    ax.scatter(
        best_idx[1], best_idx[0], best_val + 0.005,
        color='#f1c40f',
        s=200,
        edgecolor='white',
        linewidth=2,
        label=f'🌟 Best: {best_val:.4f}'
    )
    ax.legend(loc='upper right', facecolor='#0a0a1a', edgecolor='white')
    
    # Animation function
    def update(frame):
        """Update viewing angle for animation."""
        ax.view_init(elev=20, azim=frame)
        return ax,
    
    total_frames = duration * fps
    anim = FuncAnimation(
        fig, update,
        frames=np.linspace(0, 360, total_frames),
        interval=1000/fps,
        blit=False
    )
    
    # Save as GIF
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '3d_animated_surface.gif')
    
    print(f"\nRendering animation... ({total_frames} frames)")
    anim.save(
        output_path,
        writer='pillow',
        fps=fps // 2,
        dpi=100
    )
    print(f"Saved: {output_path}")
    
    plt.close()
    
    return output_path


def create_animated_3d_scatter(
    predictions_path: str,
    output_dir: str = 'outputs/figures',
    duration: int = 8,
    fps: int = 30
) -> str:
    """
    Create animated 3D scatter plot (Actual vs Predicted Volatility).
    
    Args:
        predictions_path: Path to predictions.csv
        output_dir: Output directory
        duration: Video duration in seconds
        fps: Frames per second
    
    Returns:
        Path to GIF file
    """
    import pandas as pd
    from matplotlib.animation import FuncAnimation
    
    if not os.path.exists(predictions_path):
        print(f"Warning: {predictions_path} not found. Skipping 3D scatter.")
        return None
    
    # Load predictions
    df = pd.read_csv(predictions_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Use first ticker
    tickers = df['ticker'].unique()
    df = df[df['ticker'] == tickers[0]].copy()
    df = df.sort_values('date')
    
    # Calculate error
    df['error'] = np.abs(df['actual_vol'] - df['predicted_vol'])
    df['date_num'] = (df['date'] - df['date'].min()).dt.days.values
    
    # Dark style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    
    # Scatter plot
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    norm = Normalize(vmin=df['error'].min(), vmax=df['error'].max())
    cmap = cm.coolwarm
    
    scatter = ax.scatter(
        df['predicted_vol'].values,
        df['actual_vol'].values,
        df['date_num'].values,
        c=df['error'].values,
        cmap=cmap,
        s=25,
        alpha=0.7,
        edgecolors='none'
    )
    
    # Perfect prediction line
    min_val = min(df['predicted_vol'].min(), df['actual_vol'].min())
    max_val = max(df['predicted_vol'].max(), df['actual_vol'].max())
    x_line = np.linspace(min_val, max_val, 50)
    y_line = x_line
    z_line = np.ones_like(x_line) * df['date_num'].median()
    ax.plot(x_line, y_line, z_line, color='#f1c40f', linestyle='--', linewidth=2)
    
    # Labels
    ax.set_xlabel('Predicted Volatility', fontweight='bold', labelpad=15, color='white')
    ax.set_ylabel('Actual Volatility', fontweight='bold', labelpad=15, color='white')
    ax.set_zlabel('Time (Days)', fontweight='bold', labelpad=15, color='white')
    ax.set_title('📊 Forecast Accuracy: Actual vs Predicted', 
                 fontweight='bold', fontsize=20, pad=25, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    
    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Forecast Error', fontweight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Animation
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return ax,
    
    total_frames = duration * fps
    anim = FuncAnimation(
        fig, update,
        frames=np.linspace(0, 360, total_frames),
        interval=1000/fps,
        blit=False
    )
    
    # Save as GIF
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '3d_animated_scatter.gif')
    
    print(f"\nRendering 3D scatter animation... ({total_frames} frames)")
    anim.save(
        output_path,
        writer='pillow',
        fps=fps // 2,
        dpi=100
    )
    print(f"Saved: {output_path}")
    
    plt.close()
    
    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("ANIMATED 3D VISUALIZATIONS FOR LINKEDIN")
    print("=" * 60)
    
    # Load data
    data = load_data('outputs/tables')
    
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating animated 3D visualizations...")
    print("-" * 40)
    
    # 1. Animated 3D Surface (RMSE by Model × Feature Set)
    if data['model_results'] is not None and len(data['model_results']) > 0:
        create_animated_3d_surface(
            data['model_results'],
            output_dir,
            duration=8,
            fps=30
        )
    
    # 2. Animated 3D Scatter (Actual vs Predicted)
    predictions_path = os.path.join('outputs/tables', 'predictions.csv')
    if os.path.exists(predictions_path):
        create_animated_3d_scatter(
            predictions_path,
            output_dir,
            duration=8,
            fps=30
        )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - 3d_animated_surface.gif")
    print("  - 3d_animated_scatter.gif")
    print("\n📌 LinkedIn Post Tips:")
    print("  1. Upload GIF directly to LinkedIn")
    print("  2. Caption: 'How I used ML to forecast volatility...'")
    print("  3. Tag relevant hashtags: #DataScience #QuantFinance #MachineLearning")
    print("=" * 60)


if __name__ == "__main__":
    main()