"""
Evaluation module for volatility forecasting.

This module implements all evaluation metrics, statistical tests,
and visualizations for model assessment including:
- Error metrics (RMSE, MAE, tail error)
- Mincer-Zarnowitz regression (unbiasedness test)
- Diebold-Mariano test (forecast comparison)
- Volatility cone visualization
- Prediction vs actual scatter plots
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from src import config


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_tail_error(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 95) -> float:
    """
    Compute tail error at a given percentile of absolute errors.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        percentile: Percentile threshold (e.g., 95 for 95th percentile).

    Returns:
        The percentile value of absolute errors.
    """
    abs_errors = np.abs(y_true - y_pred)
    return np.percentile(abs_errors, percentile)


def compute_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute all error metrics in one function.

    Returns:
        Dictionary with rmse, mae, tail_error_95, tail_error_99.
    """
    return {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'tail_error_95': compute_tail_error(y_true, y_pred, 95),
        'tail_error_99': compute_tail_error(y_true, y_pred, 99),
    }


def mincer_zarnowitz_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Perform Mincer-Zarnowitz regression for forecast unbiasedness.

    Regresses actual on predicted: actual = alpha + beta * predicted + error
    Tests joint null: alpha = 0 and beta = 1

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with:
            - alpha: Intercept
            - beta: Slope coefficient
            - alpha_se: Standard error of alpha
            - beta_se: Standard error of beta
            - f_stat: F-statistic for joint test
            - f_pvalue: P-value of joint test
            - r_squared: R-squared of regression
            - n: Number of observations
    """
    # Add constant for intercept
    X = sm.add_constant(y_pred)

    # Handle NaN or inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    X_clean = X[mask]

    if len(y_true_clean) < 3:
        warnings.warn("Insufficient data for Mincer-Zarnowitz regression")
        return {
            'alpha': np.nan,
            'beta': np.nan,
            'alpha_se': np.nan,
            'beta_se': np.nan,
            'f_stat': np.nan,
            'f_pvalue': np.nan,
            'r_squared': np.nan,
            'n': len(y_true_clean),
        }

    try:
        model = OLS(y_true_clean, X_clean).fit()

        # Test joint hypothesis: alpha = 0, beta = 1
        r_matrix = np.array([[1, 0], [0, 1]])
        q_matrix = np.array([0, 1])
        f_test = model.f_test((r_matrix, q_matrix))

        return {
            'alpha': model.params[0],
            'beta': model.params[1],
            'alpha_se': model.bse[0],
            'beta_se': model.bse[1],
            'f_stat': f_test.fvalue,
            'f_pvalue': f_test.pvalue,
            'r_squared': model.rsquared,
            'n': len(y_true_clean),
        }
    except Exception as e:
        warnings.warn(f"Mincer-Zarnowitz regression failed: {e}")
        return {
            'alpha': np.nan,
            'beta': np.nan,
            'alpha_se': np.nan,
            'beta_se': np.nan,
            'f_stat': np.nan,
            'f_pvalue': np.nan,
            'r_squared': np.nan,
            'n': len(y_true_clean),
        }


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    loss_func: str = 'mse'
) -> Dict[str, float]:
    """
    Perform Diebold-Mariano test comparing two forecasts.

    Tests the null hypothesis that both forecasts have equal predictive accuracy.

    Args:
        y_true: Actual values.
        y_pred1: Forecast 1 predictions.
        y_pred2: Forecast 2 predictions.
        h: Forecast horizon (used for autocorrelation adjustment).
        loss_func: 'mse' or 'mae'.

    Returns:
        Dictionary with dm_stat and p_value.
    """
    # Handle NaN/inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred1) & np.isfinite(y_pred2)
    y_true_clean = y_true[mask]
    y_pred1_clean = y_pred1[mask]
    y_pred2_clean = y_pred2[mask]

    if len(y_true_clean) < 10:
        warnings.warn("Insufficient data for Diebold-Mariano test")
        return {'dm_stat': np.nan, 'p_value': np.nan}

    # Compute loss differential
    if loss_func == 'mse':
        loss1 = (y_true_clean - y_pred1_clean) ** 2
        loss2 = (y_true_clean - y_pred2_clean) ** 2
    elif loss_func == 'mae':
        loss1 = np.abs(y_true_clean - y_pred1_clean)
        loss2 = np.abs(y_true_clean - y_pred2_clean)
    else:
        raise ValueError("loss_func must be 'mse' or 'mae'")

    d = loss1 - loss2
    d_mean = np.mean(d)

    # Compute variance of d with autocorrelation adjustment
    n = len(d)
    gamma_0 = np.var(d, ddof=1)

    if h > 1:
        # Estimate autocorrelation up to lag h-1
        gamma = [np.corrcoef(d[:-i], d[i:])[0, 1] * gamma_0 for i in range(1, h)]
        gamma_sum = sum(gamma)
        var_d = gamma_0 + 2 * gamma_sum
    else:
        var_d = gamma_0

    if var_d <= 0 or np.isnan(var_d):
        return {'dm_stat': np.nan, 'p_value': np.nan}

    dm_stat = d_mean / np.sqrt(var_d / n)

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {'dm_stat': dm_stat, 'p_value': p_value}


def ramsey_reset_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    powers: List[int] = [2, 3]
) -> Dict[str, float]:
    """
    Perform Ramsey's RESET test for model misspecification.

    Tests for non-linearity by adding powers of fitted values to the regression.

    Args:
        y_true: Actual values.
        y_pred: Predicted values from the original model.
        powers: Powers of fitted values to include (e.g., [2, 3] for squared and cubed).

    Returns:
        Dictionary with f_stat and p_value.
    """
    # Handle NaN/inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 10:
        warnings.warn("Insufficient data for RESET test")
        return {'f_stat': np.nan, 'p_value': np.nan}

    try:
        # Build augmented regression
        X_base = sm.add_constant(y_pred_clean)
        X_aug = X_base.copy()

        for power in powers:
            X_aug[f'y_pred_{power}'] = y_pred_clean ** power

        # Full model
        model_full = OLS(y_true_clean, X_aug).fit()

        # Restricted model (base only)
        model_restricted = OLS(y_true_clean, X_base).fit()

        # F-test for joint significance of added terms
        df_diff = len(powers)
        f_stat = ((model_restricted.ssr - model_full.ssr) / df_diff) / (model_full.ssr / model_full.df_resid)
        p_value = 1 - stats.f.cdf(f_stat, df_diff, model_full.df_resid)

        return {'f_stat': f_stat, 'p_value': p_value}
    except Exception as e:
        warnings.warn(f"RESET test failed: {e}")
        return {'f_stat': np.nan, 'p_value': np.nan}


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    feature_set: str,
    dates: Optional[pd.DatetimeIndex] = None,
    tickers: Optional[List[str]] = None
) -> Dict:
    """
    Complete forecast evaluation with all metrics and tests.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        model_name: Name of the model.
        feature_set: Feature set used.
        dates: Optional dates for time-series analysis.
        tickers: Optional tickers for cross-sectional analysis.

    Returns:
        Dictionary with all evaluation metrics.
    """
    # Error metrics
    error_metrics = compute_error_metrics(y_true, y_pred)

    # Mincer-Zarnowitz
    mz_results = mincer_zarnowitz_regression(y_true, y_pred)

    # RESET test for non-linearity
    reset_results = ramsey_reset_test(y_true, y_pred)

    return {
        'model': model_name,
        'feature_set': feature_set,
        'n_samples': len(y_true),
        **error_metrics,
        **mz_results,
        'reset_f_stat': reset_results['f_stat'],
        'reset_p_value': reset_results['p_value'],
    }


def evaluate_all_predictions(
    results_df: pd.DataFrame,
    y_true_series: pd.Series,
    y_pred_dict: Dict[str, np.ndarray],
    feature_set_dict: Dict[str, str]
) -> pd.DataFrame:
    """
    Evaluate all predictions from multiple models.

    Args:
        results_df: DataFrame with evaluation results to append to.
        y_true_series: Series of actual values.
        y_pred_dict: Dictionary mapping model_name -> predictions.
        feature_set_dict: Dictionary mapping model_name -> feature_set.

    Returns:
        DataFrame with evaluation results for all models.
    """
    all_eval_results = []

    for model_name, y_pred in y_pred_dict.items():
        feature_set = feature_set_dict.get(model_name, 'unknown')
        eval_result = evaluate_forecast(
            y_true=y_true_series.values,
            y_pred=y_pred,
            model_name=model_name,
            feature_set=feature_set
        )
        all_eval_results.append(eval_result)

    eval_df = pd.DataFrame(all_eval_results)

    if results_df is not None:
        eval_df = pd.concat([results_df, eval_df], ignore_index=True)

    return eval_df


def plot_volatility_cone(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str = "Volatility Cone",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot volatility cone showing predicted vs actual volatility with confidence bands.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        dates: Dates for x-axis.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by date
    idx = np.argsort(dates)
    dates_sorted = dates[idx]
    y_true_sorted = y_true[idx]
    y_pred_sorted = y_pred[idx]

    # Calculate confidence bands based on RMSE
    rmse = compute_rmse(y_true_sorted, y_pred_sorted)
    upper_band = y_pred_sorted + 1.96 * rmse
    lower_band = y_pred_sorted - 1.96 * rmse

    # Plot actual and predicted
    ax.plot(dates_sorted, y_true_sorted, label='Actual Volatility', color='black', linewidth=1.5)
    ax.plot(dates_sorted, y_pred_sorted, label='Predicted Volatility', color='blue', linewidth=1.5)

    # Confidence bands
    ax.fill_between(
        dates_sorted,
        lower_band,
        upper_band,
        color='blue',
        alpha=0.15,
        label='95% Confidence Band (1.96 × RMSE)'
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


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual Volatility",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot scatter of predicted vs actual with 45-degree line.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_pred, y_true, alpha=0.5, s=10)

    # 45-degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add regression line
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
        x_line = np.linspace(min_val, max_val, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b-', linewidth=1, label=f'Regression Line (R² = {r_value**2:.3f})')
    except Exception:
        pass

    ax.set_xlabel('Predicted Volatility')
    ax.set_ylabel('Actual Volatility')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast Error Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot distribution of forecast errors (y_true - y_pred).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    errors = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram with KDE
    sns.histplot(errors, kde=True, ax=ax1, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Forecast Error (Actual - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')

    # Q-Q plot
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')

    fig.suptitle(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'rmse',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison of models across feature sets for a given metric.

    Args:
        results_df: Results DataFrame from evaluate_all_predictions().
        metric: Metric to plot ('rmse', 'mae', 'tail_error_95', 'mz_beta').
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if title is None:
        title = f"Model Comparison: {metric.upper()}"

    # Pivot for bar plot
    pivot_df = results_df.pivot_table(
        index='model',
        columns='feature_set',
        values=metric,
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=figsize)
    pivot_df.plot(kind='bar', ax=ax)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mz_beta_by_model(
    results_df: pd.DataFrame,
    title: str = "Mincer-Zarnowitz Beta by Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot Mincer-Zarnowitz beta coefficients for all models.

    Includes a horizontal line at beta = 1 (perfect calibration).

    Args:
        results_df: Results DataFrame.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    pivot_df = results_df.pivot_table(
        index='model',
        columns='feature_set',
        values='mz_beta',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=figsize)
    pivot_df.plot(kind='bar', ax=ax)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Beta = 1 (Perfect)')
    ax.axhline(y=0.9, color='orange', linestyle=':', linewidth=1.5, label='Beta = 0.9 (Threshold)')

    ax.set_xlabel('Model')
    ax.set_ylabel('Mincer-Zarnowitz Beta')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_evaluation_report(
    results_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_dict: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    output_dir: str = 'outputs'
) -> None:
    """
    Generate complete evaluation report with all tables and figures.

    Args:
        results_df: Results DataFrame.
        y_true: Actual values (for main model).
        y_pred_dict: Dictionary of predictions.
        dates: Dates for plotting.
        output_dir: Output directory.
    """
    # Create output directories
    figures_dir = os.path.join(output_dir, 'figures')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Save results table
    results_df.to_csv(os.path.join(tables_dir, 'evaluation_results.csv'), index=False)

    # Create and save comparison tables
    best_models = results_df.loc[
        results_df.groupby('feature_set')['rmse'].idxmin()
    ][['feature_set', 'model', 'rmse', 'mae', 'mz_beta', 'mz_f_pvalue']]
    best_models.to_csv(os.path.join(tables_dir, 'best_models.csv'), index=False)

    # Generate plots for the best performing model
    # Find best model by RMSE
    best_idx = results_df['rmse'].idxmin()
    best_model = results_df.loc[best_idx, 'model']
    best_feature_set = results_df.loc[best_idx, 'feature_set']

    if best_model in y_pred_dict:
        y_pred_best = y_pred_dict[best_model]
        print(f"Generating plots for best model: {best_model} ({best_feature_set})")

        # Volatility cone
        plot_volatility_cone(
            y_true, y_pred_best, dates,
            title=f"Volatility Cone - {best_model} ({best_feature_set})",
            save_path=os.path.join(figures_dir, 'volatility_cone.png')
        )

        # Predicted vs actual
        plot_predicted_vs_actual(
            y_true, y_pred_best,
            title=f"Predicted vs Actual - {best_model} ({best_feature_set})",
            save_path=os.path.join(figures_dir, 'predicted_vs_actual.png')
        )

        # Error distribution
        plot_error_distribution(
            y_true, y_pred_best,
            title=f"Error Distribution - {best_model} ({best_feature_set})",
            save_path=os.path.join(figures_dir, 'error_distribution.png')
        )

    # Model comparison plots
    plot_model_comparison(
        results_df, metric='rmse',
        title="RMSE Comparison by Model and Feature Set",
        save_path=os.path.join(figures_dir, 'rmse_comparison.png')
    )

    plot_model_comparison(
        results_df, metric='mz_beta',
        title="Mincer-Zarnowitz Beta by Model and Feature Set",
        save_path=os.path.join(figures_dir, 'mz_beta_comparison.png')
    )

    plot_mz_beta_by_model(
        results_df,
        title="Mincer-Zarnowitz Beta by Model",
        save_path=os.path.join(figures_dir, 'mz_beta_by_model.png')
    )

    print(f"Evaluation report generated in {output_dir}")


# Quick test function
if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
    print("Available metrics: RMSE, MAE, Tail Error, Mincer-Zarnowitz, Diebold-Mariano, RESET test")
    print("Available plots: Volatility Cone, Predicted vs Actual, Error Distribution, Model Comparison")