"""
Feature selection module for volatility forecasting.
Analyzes correlations to identify redundant features.

Outputs:
  - Correlation heatmap images (PNG)
  - Text report with numerical correlation values (TXT)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path if running from src directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import get_feature_list
from src import config


def plot_correlation_matrix(
    feature_df: pd.DataFrame,
    feature_set: str,
    output_dir: str = 'outputs/figures'
):
    """
    Plot correlation matrix for features in a given feature set.
    Identifies highly correlated features (>0.8) for potential removal.
    """
    features = get_feature_list(feature_set)
    
    # Ensure all features exist in the dataframe
    available_features = [f for f in features if f in feature_df.columns]
    if len(available_features) < len(features):
        print(f"Warning: {len(features) - len(available_features)} features not found")
    
    corr = feature_df[available_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap='RdBu_r',
        center=0,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title(f'Feature Correlation Matrix: {feature_set}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'feature_correlation_{feature_set}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved image: {save_path}")
    plt.close()
    
    return corr


def generate_correlation_report(
    feature_df: pd.DataFrame,
    feature_set: str,
    output_dir: str = 'outputs/figures'
) -> str:
    """
    Generate text report with numerical correlation values.
    
    Returns:
        Report text as string
    """
    features = get_feature_list(feature_set)
    available_features = [f for f in features if f in feature_df.columns]
    corr = feature_df[available_features].corr()
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"CORRELATION REPORT: {feature_set.upper()}")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Features analyzed: {len(available_features)}")
    lines.append("")
    
    # Full correlation matrix
    lines.append("FULL CORRELATION MATRIX:")
    lines.append("-" * 40)
    
    # Get column names for header
    col_names = corr.columns.tolist()
    header = "  " + "  ".join([f"{name[:12]:>12}" for name in col_names])
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    
    for i, row_name in enumerate(corr.index):
        row_vals = []
        for j, col_name in enumerate(corr.columns):
            val = corr.iloc[i, j]
            if i == j:
                row_vals.append(f"{val:.2f}*")
            else:
                row_vals.append(f"{val:.2f}")
        lines.append(f"  {row_name[:12]:>12} " + "  ".join(row_vals))
    lines.append("")
    
    # High correlations (>0.7)
    lines.append("HIGH CORRELATIONS (>0.7):")
    lines.append("-" * 40)
    found_high = False
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if abs(corr_val) > 0.7:
                found_high = True
                lines.append(f"  {corr.columns[i]:20s} <-> {corr.columns[j]:20s} : {corr_val:.3f}")
    if not found_high:
        lines.append("  None found")
    lines.append("")
    
    # Very high correlations (>0.8) - candidates for removal
    lines.append("REDUNDANT FEATURES (>0.8) - CANDIDATES FOR REMOVAL:")
    lines.append("-" * 40)
    found_redundant = False
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if abs(corr_val) > 0.8:
                found_redundant = True
                lines.append(f"  {corr.columns[i]:20s} <-> {corr.columns[j]:20s} : {corr_val:.3f}")
    if not found_redundant:
        lines.append("  None found (>0.8 threshold)")
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY STATISTICS:")
    lines.append("-" * 40)
    corr_flat = corr.values[np.triu_indices_from(corr.values, k=1)]
    if len(corr_flat) > 0:
        lines.append(f"  Mean correlation: {np.mean(corr_flat):.3f}")
        lines.append(f"  Max correlation:  {np.max(np.abs(corr_flat)):.3f}")
        lines.append(f"  Min correlation:  {np.min(corr_flat):.3f}")
        lines.append(f"  Std correlation:  {np.std(corr_flat):.3f}")
    lines.append("")
    
    # Recommendations
    lines.append("RECOMMENDATIONS:")
    lines.append("-" * 40)
    redundant_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if abs(corr_val) > 0.8:
                redundant_pairs.append((corr.columns[i], corr.columns[j], corr_val))
    
    if redundant_pairs:
        lines.append("  Consider removing one feature from each pair:")
        for f1, f2, val in redundant_pairs:
            lines.append(f"    - Remove either '{f1}' or '{f2}' ({val:.3f})")
    else:
        lines.append("  No redundant features identified. Keep all features.")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def analyze_all_feature_sets(
    feature_df: pd.DataFrame,
    output_dir: str = 'outputs/figures'
) -> None:
    """
    Run correlation analysis for all feature sets.
    Saves both images and text reports.
    """
    feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']
    
    # Collect all reports
    all_reports = []
    all_reports.append("=" * 70)
    all_reports.append("FEATURE CORRELATION ANALYSIS - COMPLETE REPORT")
    all_reports.append("=" * 70)
    all_reports.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_reports.append(f"Feature matrix shape: {feature_df.shape}")
    all_reports.append("")
    
    for feature_set in feature_sets:
        print(f"\n{'-'*40}")
        print(f"Analyzing: {feature_set.upper()}")
        print(f"{'-'*40}")
        
        # Generate correlation matrix and plot
        corr = plot_correlation_matrix(feature_df, feature_set, output_dir)
        
        # Generate report
        report = generate_correlation_report(feature_df, feature_set, output_dir)
        all_reports.append(report)
        
        # Save individual report
        report_path = os.path.join(output_dir, f'correlation_report_{feature_set}.txt')
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved report: {report_path}")
    
    # Save combined report
    combined_report = "\n".join(all_reports)
    combined_path = os.path.join(output_dir, 'correlation_report_ALL.txt')
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write(combined_report)
    print(f"\nSaved combined report: {combined_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("  Images: feature_correlation_*.png")
    print("  Reports: correlation_report_*.txt")


def get_redundant_features(
    feature_df: pd.DataFrame,
    feature_set: str,
    threshold: float = 0.8
) -> list:
    """
    Identify redundant features (correlation > threshold).
    Returns list of features to remove.
    """
    features = get_feature_list(feature_set)
    available_features = [f for f in features if f in feature_df.columns]
    corr = feature_df[available_features].corr()
    
    to_remove = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > threshold:
                # Remove the second feature (simple heuristic)
                to_remove.add(corr.columns[j])
    
    return list(to_remove)


def main():
    """Main entry point for correlation analysis."""
    print("=" * 60)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Load feature matrix
    feature_path = os.path.join(config.PROCESSED_DATA_PATH, config.PROCESSED_FILENAME)
    if not os.path.exists(feature_path):
        print(f"Error: Feature matrix not found at {feature_path}")
        print("Run the pipeline first: python run_pipeline.py")
        return
    
    feature_df = pd.read_parquet(feature_path)
    print(f"Loaded feature matrix: {feature_df.shape}")
    print(f"  Rows: {feature_df.shape[0]:,}")
    print(f"  Columns: {feature_df.shape[1]}")
    print("")
    
    # Run analysis
    analyze_all_feature_sets(feature_df, config.FIGURES_PATH)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()