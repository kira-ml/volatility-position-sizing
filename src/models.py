"""
Model training and evaluation module for volatility forecasting.

This module implements all baseline and advanced models, and evaluates them
across all feature sets in a single run for fair comparison.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
import statsmodels.api as sm

from src import config


def temporal_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    date_col: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically by date.

    Args:
        df: DataFrame with a date column.
        test_ratio: Proportion of data to use for test set (latest dates).
        date_col: Name of the date column.

    Returns:
        Tuple of (train_df, test_df).
    """
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def prepare_features(
    df: pd.DataFrame,
    feature_set: str,
    scale: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[StandardScaler], Optional[LabelEncoder]]:
    """
    Prepare features and target for modeling.

    Args:
        df: Feature matrix with 'date', 'ticker', 'target', and features.
        feature_set: Which feature set to use ('baseline_1', 'baseline_2', 'baseline_3', 'advanced').
        scale: Whether to scale features using StandardScaler.

    Returns:
        Tuple of (X, y, scaler, label_encoder).
    """
    from src.features import get_feature_list

    # Get feature list for this feature set
    feature_cols = get_feature_list(feature_set)

    # Separate features and target
    X = df[feature_cols].copy()
    y = df['target'].copy()

    # Encode ticker as categorical feature
    if 'ticker' in df.columns:
        le = LabelEncoder()
        ticker_encoded = le.fit_transform(df['ticker'])
        X['ticker_encoded'] = ticker_encoded
    else:
        le = None

    # Scale features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y, scaler, le


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    feature_set: str
) -> Dict:
    """
    Compute all evaluation metrics for a set of predictions.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        model_name: Name of the model.
        feature_set: Feature set used.

    Returns:
        Dictionary of metrics.
    """
    # Basic error metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Tail error (95th percentile of absolute error)
    abs_errors = np.abs(y_true - y_pred)
    tail_error = np.percentile(abs_errors, 95)

    # Mincer-Zarnowitz regression
    # Regress actual on predicted: actual = alpha + beta * predicted + error
    X_mz = sm.add_constant(y_pred)
    try:
        mz_model = sm.OLS(y_true, X_mz).fit()
        alpha = mz_model.params.iloc[0]
        beta = mz_model.params.iloc[1]
        alpha_se = mz_model.bse.iloc[0]
        beta_se = mz_model.bse.iloc[1]

        # F-test for joint null: alpha = 0, beta = 1
        r_matrix = np.array([[1, 0], [0, 1]])
        q_matrix = np.array([0, 1])
        f_test = mz_model.f_test((r_matrix, q_matrix))
        f_stat = f_test.fvalue
        f_pvalue = f_test.pvalue
        r_squared = mz_model.rsquared
    except Exception:
        # Fallback if regression fails
        alpha = np.nan
        beta = np.nan
        alpha_se = np.nan
        beta_se = np.nan
        f_stat = np.nan
        f_pvalue = np.nan
        r_squared = np.nan

    return {
        'model': model_name,
        'feature_set': feature_set,
        'rmse': rmse,
        'mae': mae,
        'tail_error_95': tail_error,
        'mz_alpha': alpha,
        'mz_beta': beta,
        'mz_alpha_se': alpha_se,
        'mz_beta_se': beta_se,
        'mz_f_stat': f_stat,
        'mz_f_pvalue': f_pvalue,
        'mz_r_squared': r_squared,
        'n_samples': len(y_true),
    }


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = None
) -> Tuple[np.ndarray, Ridge]:
    """
    Train Ridge regression model with optional hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        alpha: Regularization strength (if None, perform grid search).

    Returns:
        Tuple of (predictions, trained_model).
    """
    if alpha is None:
        # Simple grid search for alpha
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge = GridSearchCV(
            Ridge(random_state=config.RANDOM_SEED),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        ridge.fit(X_train, y_train)
        best_alpha = ridge.best_params_['alpha']
        model = ridge.best_estimator_
        print(f"  Ridge best alpha: {best_alpha}")
    else:
        model = Ridge(alpha=alpha, random_state=config.RANDOM_SEED)
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions, model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 20
) -> Tuple[np.ndarray, RandomForestRegressor]:
    """
    Train Random Forest model with sensible defaults.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split a node.

    Returns:
        Tuple of (predictions, trained_model).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    num_leaves: int = 31,
    min_child_samples: int = 20
) -> Tuple[np.ndarray, lgb.LGBMRegressor]:
    """
    Train LightGBM model with sensible defaults.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage.
        max_depth: Maximum tree depth.
        num_leaves: Maximum number of leaves.
        min_child_samples: Minimum data points in a leaf.

    Returns:
        Tuple of (predictions, trained_model).
    """
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model


def run_model_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: str,
    scale: bool = True,
    ridge_alpha: float = None
) -> List[Dict]:
    """
    Run all models on a single feature set.

    Args:
        train_df: Training data (features + target).
        test_df: Test data (features + target).
        feature_set: Feature set to use.
        scale: Whether to scale features.
        ridge_alpha: Ridge alpha (None for grid search).

    Returns:
        List of result dictionaries for each model.
    """
    results = []

    # Prepare features
    X_train, y_train, scaler, le = prepare_features(train_df, feature_set, scale=scale)
    X_test, y_test, _, _ = prepare_features(test_df, feature_set, scale=scale)

    # For testing, use the same scaler and encoder
    if scaler is not None:
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    # Baseline 1: Rolling Historical Volatility (only for baseline_1 feature set)
    if feature_set == 'baseline_1' and 'rolling_vol_21' in X_test.columns:
        rolling_preds = X_test['rolling_vol_21'].values
        results.append(evaluate_predictions(
            y_test, rolling_preds, 'Rolling_21d', feature_set
        ))

    # Baseline 2: EWMA (only for baseline_2 feature set)
    if feature_set == 'baseline_2' and 'ewma_vol_94' in X_test.columns:
        ewma_preds = X_test['ewma_vol_94'].values
        results.append(evaluate_predictions(
            y_test, ewma_preds, 'EWMA_0.94', feature_set
        ))

    # Baseline 3: Ridge Regression (all feature sets)
    ridge_preds, ridge_model = train_ridge(X_train, y_train, X_test, alpha=ridge_alpha)
    results.append(evaluate_predictions(
        y_test, ridge_preds, 'Ridge', feature_set
    ))

    # Advanced: Random Forest (all feature sets)
    rf_preds, rf_model = train_random_forest(X_train, y_train, X_test)
    results.append(evaluate_predictions(
        y_test, rf_preds, 'RandomForest', feature_set
    ))

    # Advanced: LightGBM (all feature sets)
    lgb_preds, lgb_model = train_lightgbm(X_train, y_train, X_test)
    results.append(evaluate_predictions(
        y_test, lgb_preds, 'LightGBM', feature_set
    ))

    # Advanced: Ensemble (average of Random Forest and LightGBM)
    ensemble_preds = (rf_preds + lgb_preds) / 2
    results.append(evaluate_predictions(
        y_test, ensemble_preds, 'Ensemble_RF_LGB', feature_set
    ))

    return results


def run_all_experiments(
    feature_df: pd.DataFrame,
    feature_sets: Optional[List[str]] = None,
    test_ratio: float = 0.2,
    scale: bool = True,
    ridge_alpha: float = None
) -> pd.DataFrame:
    """
    Run all models across all feature sets and return comparison results.

    This is the main entry point for model experimentation.

    Args:
        feature_df: Full feature matrix from features.py.
        feature_sets: List of feature sets to test (None = all).
        test_ratio: Proportion of data for test set.
        scale: Whether to scale features.
        ridge_alpha: Ridge alpha (None for grid search).

    Returns:
        DataFrame with results for all (model, feature_set) combinations.
    """
    if feature_sets is None:
        feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']

    # Split data chronologically
    train_df, test_df = temporal_train_test_split(feature_df, test_ratio=test_ratio)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    all_results = []

    for feature_set in feature_sets:
        print(f"\n{'='*50}")
        print(f"Running experiments for feature set: {feature_set.upper()}")
        print(f"{'='*50}")

        results = run_model_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_set=feature_set,
            scale=scale,
            ridge_alpha=ridge_alpha
        )
        all_results.extend(results)

        # Print summary for this feature set
        results_df = pd.DataFrame(results)
        print(f"\nResults for {feature_set}:")
        print(results_df[['model', 'rmse', 'mae', 'mz_beta', 'mz_f_pvalue']].to_string(index=False))

    # Combine all results
    final_df = pd.DataFrame(all_results)

    return final_df


def create_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table for easy comparison across models and feature sets.

    Args:
        results_df: Results DataFrame from run_all_experiments().

    Returns:
        Pivot table with models as rows and feature sets as columns.
    """
    # Pivot for RMSE
    rmse_pivot = results_df.pivot_table(
        index='model',
        columns='feature_set',
        values='rmse',
        aggfunc='first'
    ).round(4)

    # Pivot for Mincer-Zarnowitz beta
    beta_pivot = results_df.pivot_table(
        index='model',
        columns='feature_set',
        values='mz_beta',
        aggfunc='first'
    ).round(4)

    # Pivot for Mincer-Zarnowitz p-value
    pvalue_pivot = results_df.pivot_table(
        index='model',
        columns='feature_set',
        values='mz_f_pvalue',
        aggfunc='first'
    ).round(4)

    return rmse_pivot, beta_pivot, pvalue_pivot


def get_best_model_per_feature_set(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the best model for each feature set based on RMSE.

    Args:
        results_df: Results DataFrame.

    Returns:
        DataFrame with best model per feature set.
    """
    best_models = results_df.loc[
        results_df.groupby('feature_set')['rmse'].idxmin()
    ][['feature_set', 'model', 'rmse', 'mae', 'mz_beta', 'mz_f_pvalue']]
    return best_models


def get_ridge_feature_importance(
    feature_df: pd.DataFrame,
    feature_set: str = 'baseline_3'
) -> pd.DataFrame:
    """
    Extract and display Ridge regression feature importance for a given feature set.

    This is a convenience function for interpretability.

    Args:
        feature_df: Full feature matrix.
        feature_set: Which feature set to use.

    Returns:
        DataFrame with features and their Ridge coefficients.
    """
    from src.features import get_feature_list

    # Split data
    train_df, test_df = temporal_train_test_split(feature_df)
    feature_cols = get_feature_list(feature_set)

    # Prepare features
    X_train, y_train, scaler, le = prepare_features(train_df, feature_set, scale=True)

    # Train Ridge
    ridge = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
    ridge.fit(X_train, y_train)

    # Get coefficients
    coefficients = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': ridge.coef_
    }).sort_values('coefficient', ascending=False)

    return coefficients


# Quick test function
if __name__ == "__main__":
    print("Model training module loaded successfully.")
    print("Available feature sets: baseline_1, baseline_2, baseline_3, advanced")
    print("Models: Rolling_21d, EWMA_0.94, Ridge, RandomForest, LightGBM, Ensemble_RF_LGB")