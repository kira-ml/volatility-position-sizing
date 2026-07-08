#!/usr/bin/env python
"""
Animated 3D Visualizations for LinkedIn/Instagram.

Creates rotating 3D surface plots exported as GIFs.
Perfect for social media engagement hooks.

Usage:
    python src/visualize_3d_animated_gif.py
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

from src.models import load_model, load_scaler
from src.features import get_feature_list


def create_animated_rmse_surface(
    model_results: pd.DataFrame,
    output_dir: str = 'outputs/figures',
    duration: int = 8,
    fps: int = 30
) -> str:
    """
    Animated 3D RMSE Surface: Model × Feature Set.
    """
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
    
    # Dark style
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    
    # Surface
    colors_flat = Z.flatten()
    norm = Normalize(vmin=colors_flat.min(), vmax=colors_flat.max())
    cmap = cm.plasma
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        rstride=1,
        cstride=1
    )
    
    ax.set_xlabel('Model', fontweight='bold', labelpad=15, color='white')
    ax.set_ylabel('Feature Set', fontweight='bold', labelpad=15, color='white')
    ax.set_zlabel('RMSE', fontweight='bold', labelpad=15, color='white')
    ax.set_title('🎯 Model Performance: RMSE by Model & Feature Set', 
                 fontweight='bold', fontsize=18, pad=25, color='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Ridge', 'RF', 'LGB', 'Ensemble'], rotation=30, ha='right', color='white')
    ax.set_yticks(y)
    ax.set_yticklabels(['Base 1', 'Base 2', 'Base 3', 'Advanced'], rotation=20, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    
    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('RMSE', fontweight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Best model marker
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
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '3d_animated_rmse_surface.gif')
    
    print(f"\nRendering RMSE animation... ({total_frames} frames)")
    anim.save(
        output_path,
        writer='pillow',
        fps=fps // 2,
        dpi=100
    )
    print(f"Saved: {output_path}")
    plt.close()
    
    return output_path


def create_animated_calibration_surface(
    model_results: pd.DataFrame,
    output_dir: str = 'outputs/figures',
    duration: int = 8,
    fps: int = 30
) -> str:
    """
    Animated 3D Calibration Surface: MZ Beta vs Model vs Feature Set.
    """
    models = ['Ridge', 'RandomForest', 'LightGBM', 'Ensemble_RF_LGB']
    feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']
    
    beta_matrix = []
    for model in models:
        row = model_results[model_results['model'] == model]
        if not row.empty:
            vals = []
            for fs in feature_sets:
                val = row[row['feature_set'] == fs]['mz_beta'].values
                vals.append(val[0] if len(val) > 0 else np.nan)
            beta_matrix.append(vals)
        else:
            beta_matrix.append([np.nan] * len(feature_sets))
    
    beta_matrix = np.array(beta_matrix)
    
    x = np.arange(len(models))
    y = np.arange(len(feature_sets))
    X, Y = np.meshgrid(x, y)
    Z = beta_matrix.T
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0a0a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        rstride=1,
        cstride=1,
        vmin=0.0,
        vmax=4.0
    )
    
    # Perfect calibration plane at Z=1.0
    x_plane = np.linspace(0, len(models)-1, 10)
    y_plane = np.linspace(0, len(feature_sets)-1, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.ones_like(X_plane) * 1.0
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='#f1c40f', alpha=0.15)
    
    ax.set_xlabel('Model', fontweight='bold', labelpad=15, color='white')
    ax.set_ylabel('Feature Set', fontweight='bold', labelpad=15, color='white')
    ax.set_zlabel('MZ Beta', fontweight='bold', labelpad=15, color='white')
    ax.set_title('📊 Calibration Landscape: MZ Beta by Model & Feature Set\n(Closer to 1.0 = Better)', 
                 fontweight='bold', fontsize=18, pad=25, color='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Ridge', 'RF', 'LGB', 'Ensemble'], rotation=30, ha='right', color='white')
    ax.set_yticks(y)
    ax.set_yticklabels(['Base 1', 'Base 2', 'Base 3', 'Advanced'], rotation=20, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('MZ Beta (β)', fontweight='bold', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Animation
    def update(frame):
        ax.view_init(elev=25, azim=frame)
        return ax,
    
    total_frames = duration * fps
    anim = FuncAnimation(
        fig, update,
        frames=np.linspace(0, 360, total_frames),
        interval=1000/fps,
        blit=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '3d_animated_calibration_surface.gif')
    
    print(f"\nRendering Calibration animation... ({total_frames} frames)")
    anim.save(
        output_path,
        writer='pillow',
        fps=fps // 2,
        dpi=100
    )
    print(f"Saved: {output_path}")
    plt.close()
    
    return output_path


def create_animated_prediction_surface(
    feature_df: pd.DataFrame,
    output_dir: str = 'outputs/figures',
    duration: int = 8,
    fps: int = 30
) -> str:
    """
    Animated 3D Prediction Surface from saved model.
    """
    try:
        model = load_model('lightgbm', 'advanced')
        scaler = load_scaler('advanced')
        
        features = get_feature_list('advanced')
        all_features = features + ['ticker_encoded']
        
        # Select top 2 features by variance
        feature_df_subset = feature_df[features].copy()
        variances = feature_df_subset.var().sort_values(ascending=False)
        top_features = variances.head(2).index.tolist()
        
        print(f"  Prediction surface features: {top_features}")
        
        # Create grid
        n_points = 25
        f1_min, f1_max = feature_df[top_features[0]].quantile(0.05), feature_df[top_features[0]].quantile(0.95)
        f2_min, f2_max = feature_df[top_features[1]].quantile(0.05), feature_df[top_features[1]].quantile(0.95)
        
        f1_vals = np.linspace(f1_min, f1_max, n_points)
        f2_vals = np.linspace(f2_min, f2_max, n_points)
        F1, F2 = np.meshgrid(f1_vals, f2_vals)
        
        # Create prediction grid
        pred_grid = np.zeros_like(F1)
        for i in range(n_points):
            for j in range(n_points):
                X_sample = pd.DataFrame({
                    features[0]: [F1[i, j]],
                    features[1]: [F2[i, j]],
                })
                for feat in features[2:]:
                    X_sample[feat] = feature_df[feat].median()
                X_sample['ticker_encoded'] = 0
                X_sample = X_sample[all_features]
                
                X_scaled = scaler.transform(X_sample)
                pred_grid[i, j] = model.predict(X_scaled)[0]
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 8), facecolor='#0a0a1a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a1a')
        
        surf = ax.plot_surface(
            F1, F2, pred_grid,
            cmap=cm.plasma,
            linewidth=0,
            antialiased=True,
            alpha=0.9
        )
        
        ax.set_xlabel(top_features[0], fontweight='bold', labelpad=15, color='white')
        ax.set_ylabel(top_features[1], fontweight='bold', labelpad=15, color='white')
        ax.set_zlabel('Predicted Volatility', fontweight='bold', labelpad=15, color='white')
        ax.set_title(f'🧠 Model Prediction Surface: LightGBM\n{top_features[0]} vs {top_features[1]}', 
                     fontweight='bold', fontsize=18, pad=25, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
        cbar.set_label('Predicted Volatility', fontweight='bold', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Animation
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return ax,
        
        total_frames = duration * fps
        anim = FuncAnimation(
            fig, update,
            frames=np.linspace(0, 360, total_frames),
            interval=1000/fps,
            blit=False
        )
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, '3d_animated_prediction_surface.gif')
        
        print(f"\nRendering Prediction Surface animation... ({total_frames} frames)")
        anim.save(
            output_path,
            writer='pillow',
            fps=fps // 2,
            dpi=100
        )
        print(f"Saved: {output_path}")
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"Warning: Could not create prediction surface: {e}")
        return None


def main():
    """Generate all animated 3D visualizations."""
    print("=" * 60)
    print("ANIMATED 3D VISUALIZATIONS FOR LINKEDIN")
    print("=" * 60)
    
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    model_results_path = 'outputs/tables/model_results.csv'
    if not os.path.exists(model_results_path):
        print(f"Error: {model_results_path} not found. Run the pipeline first.")
        return
    
    model_results = pd.read_csv(model_results_path)
    print(f"Loaded model results: {len(model_results)} rows")
    
    feature_path = 'data/processed/feature_matrix.parquet'
    if os.path.exists(feature_path):
        feature_df = pd.read_parquet(feature_path)
        print(f"Loaded feature data: {feature_df.shape}")
    else:
        print(f"Warning: {feature_path} not found.")
        feature_df = None
    
    print("\nGenerating animated 3D GIFs...")
    print("-" * 40)
    
    # 1. Animated RMSE Surface
    create_animated_rmse_surface(model_results, output_dir)
    
    # 2. Animated Calibration Surface
    create_animated_calibration_surface(model_results, output_dir)
    
    # 3. Animated Prediction Surface (if feature data available)
    if feature_df is not None:
        create_animated_prediction_surface(feature_df, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Animated GIFs generated:")
    print("  - 3d_animated_rmse_surface.gif")
    print("  - 3d_animated_calibration_surface.gif")
    print("  - 3d_animated_prediction_surface.gif")
    print("\n📌 LinkedIn Post Tips:")
    print("  1. Upload GIF directly to LinkedIn")
    print("  2. Caption: 'How I used ML to forecast volatility...'")
    print("  3. Hashtags: #DataScience #QuantFinance #MachineLearning")
    print("=" * 60)


if __name__ == "__main__":
    main()