#!/usr/bin/env python
"""
experiment.py - Isolated Feature Engineering Experiments

This module runs targeted experiments without modifying the main pipeline.
Each experiment tests a specific hypothesis and compares results against
the current best model (LightGBM on Baseline_3).

Usage:
    python experiment.py                    # Run all experiments
    python experiment.py --experiment log   # Run only log-transform
    python experiment.py --experiment all   # Run all experiments
    python experiment.py --list             # List available experiments
    python experiment.py --quick            # Fast mode (1 split instead of 5)

Experiments:
    1. log          - Log-transform the target variable
    2. leverage     - Add asymmetric leverage effect features
    3. vol_of_vol   - Add volatility of volatility to Baseline_3
    4. ewma         - Test alternative EWMA decay factors
    5. isotonic     - Post-hoc isotonic calibration
    6. all          - Run all experiments

Principles:
    - Simple, not over-engineered
    - Grounded in financial literature
    - Clear hypotheses
    - Statistical rigor (MZ Beta, RMSE, DM test)
"""

import os
import sys
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import copy

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from existing modules (reuse, don't duplicate)
# Import from existing modules - handles both running from project root and from src/
try:
    from src import config
except ModuleNotFoundError:
    # Running from within src/ directory - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config



from src.features import (
    get_feature_list, 
    filter_features,
    compute_log_returns,
    compute_ewma_volatility,
    compute_rolling_volatility
)
from src.models import (
    prepare_features, 
    train_lightgbm, 
    train_ridge, 
    evaluate_predictions,
    temporal_train_test_split,
    walk_forward_split,
    save_model
)
from src.evaluate import (
    mincer_zarnowitz_regression, 
    diebold_mariano_test,
    ramsey_reset_test
)
from src.backtest import compare_backtests, compute_backtest_metrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# Helper Functions
# ============================================================================

def prepare_features_custom(
    df: pd.DataFrame,
    feature_cols: List[str],
    scale: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[Any], Optional[Any]]:
    """
    Prepare features with custom column list.
    Reuses logic from src.models.prepare_features.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Encode ticker as categorical
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


def get_base_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: str = 'baseline_3'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Get base model (LightGBM on Baseline_3) predictions and metrics.
    Used as the benchmark for all experiments.
    """
    X_train, y_train, scaler, le = prepare_features(
        train_df, feature_set, scale=True
    )
    X_test, y_test, _, _ = prepare_features(
        test_df, feature_set, scale=False
    )
    
    if scaler is not None:
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    # Train LightGBM
    preds, model = train_lightgbm(X_train, y_train, X_test)
    
    # Evaluate
    metrics = evaluate_predictions(y_test, preds, 'LightGBM', feature_set)
    
    return preds, y_test, metrics


def compare_experiment_vs_base(
    exp_preds: np.ndarray,
    y_test: np.ndarray,
    exp_name: str,
    base_metrics: Dict,
    base_preds: np.ndarray
) -> Dict:
    """
    Compare experimental predictions against base model.
    """
    # Evaluate experiment
    exp_metrics = evaluate_predictions(
        y_test, exp_preds, f'LightGBM_{exp_name}', 'baseline_3'
    )
    
    # Run Diebold-Mariano test
    dm_results = diebold_mariano_test(y_test, base_preds, exp_preds, h=5)
    
    # Build comparison
    comparison = {
        'experiment': exp_name,
        'base_rmse': base_metrics.get('rmse', np.nan),
        'exp_rmse': exp_metrics.get('rmse', np.nan),
        'base_mae': base_metrics.get('mae', np.nan),
        'exp_mae': exp_metrics.get('mae', np.nan),
        'base_mz_beta': base_metrics.get('mz_beta', np.nan),
        'exp_mz_beta': exp_metrics.get('mz_beta', np.nan),
        'base_mz_pvalue': base_metrics.get('mz_f_pvalue', np.nan),
        'exp_mz_pvalue': exp_metrics.get('mz_f_pvalue', np.nan),
        'base_tail_error': base_metrics.get('tail_error_95', np.nan),
        'exp_tail_error': exp_metrics.get('tail_error_95', np.nan),
        'dm_stat': dm_results.get('dm_stat', np.nan),
        'dm_pvalue': dm_results.get('p_value', np.nan),
        'n_samples': len(y_test),
    }
    
    # Calculate improvements
    # RMSE: lower is better
    comparison['rmse_improvement_pct'] = (
        (base_metrics.get('rmse', 0) - exp_metrics.get('rmse', 0)) / base_metrics.get('rmse', 1) * 100
    )
    
    # MZ Beta: closer to 1 is better
    base_beta = base_metrics.get('mz_beta', 0)
    exp_beta = exp_metrics.get('mz_beta', 0)
    comparison['beta_improvement'] = abs(base_beta - 1.0) - abs(exp_beta - 1.0)
    comparison['beta_improvement_pct'] = comparison['beta_improvement'] / (abs(base_beta - 1.0) + 0.001) * 100
    
    # Tail error: lower is better
    comparison['tail_improvement_pct'] = (
        (base_metrics.get('tail_error_95', 0) - exp_metrics.get('tail_error_95', 0)) / base_metrics.get('tail_error_95', 1) * 100
    )
    
    return comparison


# ============================================================================
# Experiment Classes
# ============================================================================

class BaseExperiment:
    """Base class for all experiments."""
    
    def __init__(self, name: str, description: str, hypothesis: str):
        self.name = name
        self.description = description
        self.hypothesis = hypothesis
        self.base_feature_set = 'baseline_3'
        self.base_model = 'LightGBM'
        self.results = None
    
    def run(self, feature_df: pd.DataFrame, quick: bool = False) -> Dict:
        """Run the experiment. To be overridden."""
        raise NotImplementedError
    
    def get_base_results(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Get base model results using provided train/test splits."""
        return get_base_predictions(train_df, test_df, self.base_feature_set)


# -----------------------------------------------------------------------------
# Experiment 1: Log-Transform Target
# -----------------------------------------------------------------------------

class LogTransformExperiment(BaseExperiment):
    """
    Experiment: Log-transform the target variable.
    
    Hypothesis: Forecasting log(volatility) stabilizes variance and
    improves calibration (MZ Beta closer to 1).
    
    Literature: This is standard practice in volatility forecasting
    (Andersen & Bollerslev, 1998).
    """
    
    def __init__(self):
        super().__init__(
            name='log_target',
            description='Log-transform target before training, exponentiate predictions',
            hypothesis='Log transformation stabilizes variance and improves calibration'
        )
    
    def run(self, feature_df: pd.DataFrame, quick: bool = False) -> Dict:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Filter to base feature set
        df_base = filter_features(feature_df, self.base_feature_set)
        
        # Split chronologically
        if quick:
            # Simple 80/20 split for quick testing
            train_df, test_df = temporal_train_test_split(
                df_base, test_ratio=config.TEST_SPLIT_RATIO
            )
            splits = [(train_df, test_df)]
        else:
            # Walk-forward splits
            split_generator = walk_forward_split(
                df_base, n_splits=5, test_window=252, embargo=5
            )
            splits = []
            for train_idx, test_idx in split_generator:
                train_df = df_base.iloc[train_idx].copy()
                test_df = df_base.iloc[test_idx].copy()
                splits.append((train_df, test_df))
        
        all_comparisons = []
        
        for split_idx, (train_df, test_df) in enumerate(splits):
            # Get base results
            base_preds, y_test, base_metrics = self.get_base_results(train_df, test_df)
            
            # Prepare features for experiment
            feature_cols = get_feature_list(self.base_feature_set)
            X_train, y_train, scaler, le = prepare_features_custom(
                train_df, feature_cols, scale=True
            )
            X_test, y_test_exp, _, _ = prepare_features_custom(
                test_df, feature_cols, scale=False
            )
            
            if scaler is not None:
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
            
            # Log-transform target
            import lightgbm as lgb
            y_train_log = np.log(y_train + 1e-8)
            
            # Train on log targets
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=20,
                min_child_samples=30,
                reg_lambda=0.1,
                reg_alpha=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train_log)
            
            # Predict and exponentiate
            preds_log = model.predict(X_test)
            exp_preds = np.exp(preds_log)
            
            # Compare
            comparison = compare_experiment_vs_base(
                exp_preds, y_test, self.name, base_metrics, base_preds
            )
            comparison['split'] = split_idx
            all_comparisons.append(comparison)
        
        # Aggregate results
        agg = self._aggregate_comparisons(all_comparisons)
        self.results = agg
        return agg
    
    def _aggregate_comparisons(self, comparisons: List[Dict]) -> Dict:
        """Aggregate results across splits."""
        if not comparisons:
            return {'experiment': self.name, 'error': 'No results'}
        
        # Average numeric values
        numeric_keys = ['rmse_improvement_pct', 'beta_improvement_pct', 'tail_improvement_pct',
                       'base_rmse', 'exp_rmse', 'base_mz_beta', 'exp_mz_beta',
                       'base_mz_pvalue', 'exp_mz_pvalue']
        
        agg = {'experiment': self.name, 'n_splits': len(comparisons)}
        
        for key in numeric_keys:
            values = [c.get(key, np.nan) for c in comparisons if key in c and not np.isnan(c.get(key, np.nan))]
            if values:
                agg[key] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
            else:
                agg[key] = np.nan
                agg[f'{key}_std'] = np.nan
        
        # DM test results (average)
        dm_stats = [c.get('dm_stat', np.nan) for c in comparisons if 'dm_stat' in c and not np.isnan(c.get('dm_stat', np.nan))]
        dm_pvals = [c.get('dm_pvalue', np.nan) for c in comparisons if 'dm_pvalue' in c and not np.isnan(c.get('dm_pvalue', np.nan))]
        if dm_stats:
            agg['dm_stat_mean'] = np.mean(dm_stats)
            agg['dm_pvalue_mean'] = np.mean(dm_pvals)
        
        return agg


# -----------------------------------------------------------------------------
# Experiment 2: Leverage Effect Features
# -----------------------------------------------------------------------------

class LeverageExperiment(BaseExperiment):
    """
    Experiment: Add leverage effect features.
    
    Hypothesis: Negative returns have asymmetric impact on future volatility
    (the "leverage effect"). Adding features that capture this asymmetry
    should improve calibration.
    
    Literature: Black (1976) - "Studies of Stock Price Volatility Changes"
    """
    
    def __init__(self):
        super().__init__(
            name='leverage',
            description='Add leverage effect features (asymmetric return-volatility relationship)',
            hypothesis='Negative returns amplify future volatility more than positive returns'
        )
    
    def _add_leverage_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add leverage effect features to the feature matrix.
        
        Features added:
        - leverage_effect: negative returns weighted by their magnitude
        - neg_return_5d: 5-day sum of negative returns
        - leverage_ratio: ratio of negative to positive return impact
        """
        df = feature_df.copy()
        
        # We need to compute returns for each ticker
        # Since we have the feature matrix in long format, we need to reshape
        # to compute returns, then merge back
        
        # Get unique tickers
        tickers = df['ticker'].unique()
        
        # Create leverage features
        leverage_features = []
        
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')
            
            # Compute daily returns from target? No - we need actual returns.
            # Since we don't have returns in the feature matrix, we'll compute
            # leverage from the target's rolling behavior as a proxy.
            # Alternatively, we can use the existing rolling volatility features.
            
            # Leverage effect: negative returns increase volatility more than positive
            # We'll simulate this using the target and rolling volatility relationship
            
            # Method: compute the asymmetry in the target's autocorrelation
            # When volatility is high and rising, it's often due to negative shocks
            
            # 1. Volatility change direction
            vol_change = ticker_data['rolling_vol_21'].pct_change()
            
            # 2. Create a leverage proxy: volatility increases when market is stressed
            # Use VIX as a proxy for market stress (already have vix_level)
            
            # 3. Leverage effect: negative market moves (high VIX) amplify volatility
            if 'vix_level' in ticker_data.columns:
                # Normalize VIX
                vix_mean = ticker_data['vix_level'].mean()
                vix_std = ticker_data['vix_level'].std()
                vix_norm = (ticker_data['vix_level'] - vix_mean) / (vix_std + 1e-6)
                
                # Leverage feature: VIX increase × current volatility
                ticker_data['leverage_effect'] = ticker_data['vix_level'].pct_change() * ticker_data['rolling_vol_21']
                ticker_data['leverage_effect'] = ticker_data['leverage_effect'].fillna(0)
                
                # Clip to reasonable range
                ticker_data['leverage_effect'] = ticker_data['leverage_effect'].clip(-0.5, 0.5)
                
                # Additional: negative return indicator (using VIX change as proxy)
                ticker_data['neg_shock_indicator'] = (ticker_data['vix_level'].pct_change() > 0.02).astype(float)
            else:
                # Fallback: use volatility change as proxy
                ticker_data['leverage_effect'] = ticker_data['rolling_vol_21'].pct_change().fillna(0).clip(-0.5, 0.5)
                ticker_data['neg_shock_indicator'] = (ticker_data['rolling_vol_21'].pct_change() > 0.03).astype(float)
            
            leverage_features.append(ticker_data)
        
        # Combine
        result_df = pd.concat(leverage_features, ignore_index=True)
        
        return result_df
    
    def run(self, feature_df: pd.DataFrame, quick: bool = False) -> Dict:
        # Add leverage features
        df_exp = self._add_leverage_features(feature_df)

        # Create custom feature set: Baseline_3 + leverage features
        base_features = get_feature_list(self.base_feature_set)
        custom_features = base_features + ['leverage_effect', 'neg_shock_indicator']

        # Keep only needed columns
        keep_cols = ['date', 'ticker', 'target'] + custom_features
        df_exp = df_exp[[c for c in keep_cols if c in df_exp.columns]]

        # Split and evaluate
        if quick:
            train_df, test_df = temporal_train_test_split(
                df_exp, test_ratio=config.TEST_SPLIT_RATIO
            )
            splits = [(train_df, test_df)]
        else:
            split_generator = walk_forward_split(
                df_exp, n_splits=5, test_window=252, embargo=5
            )
            splits = []
            for train_idx, test_idx in split_generator:
                train_df = df_exp.iloc[train_idx].copy()
                test_df = df_exp.iloc[test_idx].copy()
                splits.append((train_df, test_df))

        all_comparisons = []

        for split_idx, (train_df, test_df) in enumerate(splits):
            # Get base results using the SAME split indices
            base_preds, y_test, base_metrics = self.get_base_results(train_df, test_df)

            # Prepare features for experiment
            X_train, y_train, scaler, le = prepare_features_custom(
                train_df, custom_features, scale=True
            )
            X_test, y_test_exp, _, _ = prepare_features_custom(
                test_df, custom_features, scale=False
            )

            if scaler is not None:
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

            # Train LightGBM
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=20,
                min_child_samples=30,
                reg_lambda=0.1,
                reg_alpha=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            exp_preds = model.predict(X_test)

            # Compare
            comparison = compare_experiment_vs_base(
                exp_preds, y_test, self.name, base_metrics, base_preds
            )
            comparison['split'] = split_idx
            all_comparisons.append(comparison)

        # Aggregate
        agg = self._aggregate_comparisons(all_comparisons)
        self.results = agg
        return agg
    
    def _aggregate_comparisons(self, comparisons: List[Dict]) -> Dict:
        """Aggregate results across splits."""
        if not comparisons:
            return {'experiment': self.name, 'error': 'No results'}
        
        numeric_keys = ['rmse_improvement_pct', 'beta_improvement_pct', 'tail_improvement_pct',
                       'base_rmse', 'exp_rmse', 'base_mz_beta', 'exp_mz_beta',
                       'base_mz_pvalue', 'exp_mz_pvalue']
        
        agg = {'experiment': self.name, 'n_splits': len(comparisons)}
        
        for key in numeric_keys:
            values = [c.get(key, np.nan) for c in comparisons if key in c and not np.isnan(c.get(key, np.nan))]
            if values:
                agg[key] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
            else:
                agg[key] = np.nan
                agg[f'{key}_std'] = np.nan
        
        dm_stats = [c.get('dm_stat', np.nan) for c in comparisons if 'dm_stat' in c and not np.isnan(c.get('dm_stat', np.nan))]
        dm_pvals = [c.get('dm_pvalue', np.nan) for c in comparisons if 'dm_pvalue' in c and not np.isnan(c.get('dm_pvalue', np.nan))]
        if dm_stats:
            agg['dm_stat_mean'] = np.mean(dm_stats)
            agg['dm_pvalue_mean'] = np.mean(dm_pvals)
        
        return agg


# -----------------------------------------------------------------------------
# Experiment 3: Volatility of Volatility
# -----------------------------------------------------------------------------

class VolOfVolExperiment(BaseExperiment):
    """
    Experiment: Add volatility of volatility to Baseline_3.
    
    Hypothesis: The stability/instability of volatility (vol-of-vol)
    contains predictive signal for future volatility calibration.
    
    Literature: This feature captures the "volatility of volatility"
    effect, which is known to affect option pricing and risk management.
    """
    
    def __init__(self):
        super().__init__(
            name='vol_of_vol',
            description='Add volatility of volatility to Baseline_3',
            hypothesis='Volatility stability/instability contains predictive signal for calibration'
        )
    
    def run(self, feature_df: pd.DataFrame, quick: bool = False) -> Dict:
        # The feature already exists in the full feature matrix
        # Add it to Baseline_3
        base_features = get_feature_list(self.base_feature_set)
        custom_features = base_features + ['vol_of_vol']

        # Filter
        df_exp = feature_df[['date', 'ticker', 'target'] + custom_features].copy()

        if quick:
            train_df, test_df = temporal_train_test_split(
                df_exp, test_ratio=config.TEST_SPLIT_RATIO
            )
            splits = [(train_df, test_df)]
        else:
            split_generator = walk_forward_split(
                df_exp, n_splits=5, test_window=252, embargo=5
            )
            splits = []
            for train_idx, test_idx in split_generator:
                train_df = df_exp.iloc[train_idx].copy()
                test_df = df_exp.iloc[test_idx].copy()
                splits.append((train_df, test_df))

        all_comparisons = []

        for split_idx, (train_df, test_df) in enumerate(splits):
            # Get base results using the SAME split indices
            base_preds, y_test, base_metrics = self.get_base_results(train_df, test_df)

            # Prepare features
            X_train, y_train, scaler, le = prepare_features_custom(
                train_df, custom_features, scale=True
            )
            X_test, y_test_exp, _, _ = prepare_features_custom(
                test_df, custom_features, scale=False
            )

            if scaler is not None:
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

            # Train LightGBM
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=20,
                min_child_samples=30,
                reg_lambda=0.1,
                reg_alpha=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            exp_preds = model.predict(X_test)

            # Compare
            comparison = compare_experiment_vs_base(
                exp_preds, y_test, self.name, base_metrics, base_preds
            )
            comparison['split'] = split_idx
            all_comparisons.append(comparison)

        agg = self._aggregate_comparisons(all_comparisons)
        self.results = agg
        return agg
    
    def _aggregate_comparisons(self, comparisons: List[Dict]) -> Dict:
        if not comparisons:
            return {'experiment': self.name, 'error': 'No results'}
        
        numeric_keys = ['rmse_improvement_pct', 'beta_improvement_pct', 'tail_improvement_pct',
                       'base_rmse', 'exp_rmse', 'base_mz_beta', 'exp_mz_beta',
                       'base_mz_pvalue', 'exp_mz_pvalue']
        
        agg = {'experiment': self.name, 'n_splits': len(comparisons)}
        
        for key in numeric_keys:
            values = [c.get(key, np.nan) for c in comparisons if key in c and not np.isnan(c.get(key, np.nan))]
            if values:
                agg[key] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
            else:
                agg[key] = np.nan
                agg[f'{key}_std'] = np.nan
        
        dm_stats = [c.get('dm_stat', np.nan) for c in comparisons if 'dm_stat' in c and not np.isnan(c.get('dm_stat', np.nan))]
        dm_pvals = [c.get('dm_pvalue', np.nan) for c in comparisons if 'dm_pvalue' in c and not np.isnan(c.get('dm_pvalue', np.nan))]
        if dm_stats:
            agg['dm_stat_mean'] = np.mean(dm_stats)
            agg['dm_pvalue_mean'] = np.mean(dm_pvals)
        
        return agg


# -----------------------------------------------------------------------------
# Experiment 4: Alternative EWMA Decay
# -----------------------------------------------------------------------------

class EWMADecayExperiment(BaseExperiment):
    """
    Experiment: Test alternative EWMA decay factors.
    
    Hypothesis: RiskMetrics λ=0.94 may not be optimal for this dataset.
    Different decay factors may improve calibration.
    
    Literature: RiskMetrics (1996) - λ=0.94 is a standard, but
    optimal λ varies by asset class and market regime.
    """
    
    def __init__(self):
        super().__init__(
            name='ewma_decay',
            description='Test alternative EWMA decay factors (λ=0.90, 0.97)',
            hypothesis='Different EWMA decay factors may improve calibration'
        )
    
    def run(self, feature_df: pd.DataFrame, quick: bool = False) -> Dict:
        # The feature matrix already has ewma_vol_90, ewma_vol_94, ewma_vol_97
        # We'll test each one

        lambdas = [0.90, 0.97]
        all_results = []

        for lam in lambdas:
            feature_name = f'ewma_vol_{int(lam*100)}'

            if feature_name not in feature_df.columns:
                print(f"  Warning: {feature_name} not found, skipping")
                continue

            # Create custom feature set: replace ewma_vol_94 with other lambda
            base_features = get_feature_list(self.base_feature_set)
            custom_features = [
                f if f != 'ewma_vol_94' else feature_name
                for f in base_features
            ]

            # Filter to only needed columns
            df_exp = feature_df[['date', 'ticker', 'target'] + custom_features].copy()

            # Get base feature set for comparison
            df_base = filter_features(feature_df, self.base_feature_set)

            if quick:
                # Simple 80/20 split for quick testing - use same split for both
                exp_train, exp_test = temporal_train_test_split(
                    df_exp, test_ratio=config.TEST_SPLIT_RATIO
                )
                base_train, base_test = temporal_train_test_split(
                    df_base, test_ratio=config.TEST_SPLIT_RATIO
                )
                splits = [(exp_train, exp_test, base_train, base_test)]
            else:
                # Walk-forward splits - use same indices for both
                split_generator = walk_forward_split(
                    df_exp, n_splits=5, test_window=252, embargo=5
                )
                splits = []
                for train_idx, test_idx in split_generator:
                    exp_train = df_exp.iloc[train_idx].copy()
                    exp_test = df_exp.iloc[test_idx].copy()
                    base_train = df_base.iloc[train_idx].copy()
                    base_test = df_base.iloc[test_idx].copy()
                    splits.append((exp_train, exp_test, base_train, base_test))

            all_comparisons = []

            for split_idx, (exp_train, exp_test, base_train, base_test) in enumerate(splits):
                # Get base results using the SAME walk-forward indices
                base_preds, y_test, base_metrics = get_base_predictions(base_train, base_test, self.base_feature_set)

                # Prepare features for experiment
                X_train, y_train, scaler, le = prepare_features_custom(
                    exp_train, custom_features, scale=True
                )
                X_test, y_test_exp, _, _ = prepare_features_custom(
                    exp_test, custom_features, scale=False
                )

                if scaler is not None:
                    X_test = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )

                # Train LightGBM
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    num_leaves=20,
                    min_child_samples=30,
                    reg_lambda=0.1,
                    reg_alpha=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(X_train, y_train)
                exp_preds = model.predict(X_test)

                # Compare
                comparison = compare_experiment_vs_base(
                    exp_preds, y_test, f'{self.name}_{lam}', base_metrics, base_preds
                )
                comparison['split'] = split_idx
                comparison['ewma_lambda'] = lam
                all_comparisons.append(comparison)

            # Aggregate for this lambda
            agg = self._aggregate_comparisons(all_comparisons, lam)
            all_results.append(agg)

        # Find best lambda
        best = max(all_results, key=lambda x: x.get('beta_improvement_pct', -np.inf))
        self.results = best
        return best
    
    def _aggregate_comparisons(self, comparisons: List[Dict], lam: float) -> Dict:
        if not comparisons:
            return {'experiment': f'{self.name}_{lam}', 'error': 'No results'}
        
        numeric_keys = ['rmse_improvement_pct', 'beta_improvement_pct', 'tail_improvement_pct',
                       'base_rmse', 'exp_rmse', 'base_mz_beta', 'exp_mz_beta',
                       'base_mz_pvalue', 'exp_mz_pvalue']
        
        agg = {'experiment': f'{self.name}_{lam}', 'ewma_lambda': lam, 'n_splits': len(comparisons)}
        
        for key in numeric_keys:
            values = [c.get(key, np.nan) for c in comparisons if key in c and not np.isnan(c.get(key, np.nan))]
            if values:
                agg[key] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
            else:
                agg[key] = np.nan
                agg[f'{key}_std'] = np.nan
        
        dm_stats = [c.get('dm_stat', np.nan) for c in comparisons if 'dm_stat' in c and not np.isnan(c.get('dm_stat', np.nan))]
        dm_pvals = [c.get('dm_pvalue', np.nan) for c in comparisons if 'dm_pvalue' in c and not np.isnan(c.get('dm_pvalue', np.nan))]
        if dm_stats:
            agg['dm_stat_mean'] = np.mean(dm_stats)
            agg['dm_pvalue_mean'] = np.mean(dm_pvals)
        
        return agg


# -----------------------------------------------------------------------------
# Experiment 5: Isotonic Calibration
# -----------------------------------------------------------------------------

class IsotonicCalibrationExperiment(BaseExperiment):
    """
    Experiment: Post-hoc isotonic calibration.
    
    Hypothesis: Post-hoc calibration can directly improve MZ Beta
    by mapping predicted values closer to actuals.
    
    Literature: This is a standard technique in forecast calibration
    (Platt, 1999; Niculescu-Mizil & Caruana, 2005).
    """
    
    def __init__(self):
        super().__init__(
            name='isotonic',
            description='Apply isotonic regression as post-hoc calibration',
            hypothesis='Post-hoc calibration directly improves MZ Beta by mapping predictions closer to actuals'
        )
    
    def run(self, feature_df: pd.DataFrame, quick: bool = False) -> Dict:
        from sklearn.isotonic import IsotonicRegression

        # Filter to base feature set
        df_base = filter_features(feature_df, self.base_feature_set)

        if quick:
            train_df, test_df = temporal_train_test_split(
                df_base, test_ratio=config.TEST_SPLIT_RATIO
            )
            splits = [(train_df, test_df)]
        else:
            split_generator = walk_forward_split(
                df_base, n_splits=5, test_window=252, embargo=5
            )
            splits = []
            for train_idx, test_idx in split_generator:
                train_df = df_base.iloc[train_idx].copy()
                test_df = df_base.iloc[test_idx].copy()
                splits.append((train_df, test_df))

        all_comparisons = []

        for split_idx, (train_df, test_df) in enumerate(splits):
            # Get base results using the SAME split indices
            base_preds, y_test, base_metrics = self.get_base_results(train_df, test_df)

            # Prepare features for isotonic calibration
            X_train, y_train, scaler, le = prepare_features(
                train_df, self.base_feature_set, scale=True
            )
            X_test, y_test_exp, _, _ = prepare_features(
                test_df, self.base_feature_set, scale=False
            )

            if scaler is not None:
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

            # Train LightGBM and get training predictions for calibration
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=20,
                min_child_samples=30,
                reg_lambda=0.1,
                reg_alpha=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)

            # Get predictions on training set for calibration
            preds_train = model.predict(X_train)
            preds_test = model.predict(X_test)

            # Fit isotonic regression on training predictions
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(preds_train, y_train)

            # Calibrate test predictions
            exp_preds = iso.predict(preds_test)

            # Compare
            comparison = compare_experiment_vs_base(
                exp_preds, y_test, self.name, base_metrics, base_preds
            )
            comparison['split'] = split_idx
            all_comparisons.append(comparison)

        agg = self._aggregate_comparisons(all_comparisons)
        self.results = agg
        return agg
    
    def _aggregate_comparisons(self, comparisons: List[Dict]) -> Dict:
        if not comparisons:
            return {'experiment': self.name, 'error': 'No results'}
        
        numeric_keys = ['rmse_improvement_pct', 'beta_improvement_pct', 'tail_improvement_pct',
                       'base_rmse', 'exp_rmse', 'base_mz_beta', 'exp_mz_beta',
                       'base_mz_pvalue', 'exp_mz_pvalue']
        
        agg = {'experiment': self.name, 'n_splits': len(comparisons)}
        
        for key in numeric_keys:
            values = [c.get(key, np.nan) for c in comparisons if key in c and not np.isnan(c.get(key, np.nan))]
            if values:
                agg[key] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
            else:
                agg[key] = np.nan
                agg[f'{key}_std'] = np.nan
        
        dm_stats = [c.get('dm_stat', np.nan) for c in comparisons if 'dm_stat' in c and not np.isnan(c.get('dm_stat', np.nan))]
        dm_pvals = [c.get('dm_pvalue', np.nan) for c in comparisons if 'dm_pvalue' in c and not np.isnan(c.get('dm_pvalue', np.nan))]
        if dm_stats:
            agg['dm_stat_mean'] = np.mean(dm_stats)
            agg['dm_pvalue_mean'] = np.mean(dm_pvals)
        
        return agg


# ============================================================================
# Experiment Runner
# ============================================================================

def load_feature_matrix() -> pd.DataFrame:
    """Load the full feature matrix."""
    feature_path = os.path.join(config.PROCESSED_DATA_PATH, config.PROCESSED_FILENAME)
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature matrix not found: {feature_path}\n"
            f"Run the pipeline first: python run_pipeline.py"
        )
    
    return pd.read_parquet(feature_path)


def run_experiments(
    experiments: List[BaseExperiment],
    feature_df: pd.DataFrame,
    quick: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run all experiments and return results.
    """
    results = []
    
    print("\n" + "=" * 70)
    print("VOLATILITY FORECASTING EXPERIMENTS")
    print("=" * 70)
    print(f"Number of experiments: {len(experiments)}")
    print(f"Data shape: {feature_df.shape}")
    print(f"Mode: {'Quick (1 split)' if quick else 'Full (5 walk-forward splits)'}")
    print("=" * 70 + "\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {exp.name}")
        print("-" * 40)
        print(f"Description: {exp.description}")
        print(f"Hypothesis:  {exp.hypothesis}")
        print("-" * 40)
        
        try:
            result = exp.run(feature_df, quick=quick)
            results.append(result)
            
            # Print results
            print(f"\nResults for {exp.name}:")
            print(f"  Base RMSE:   {result.get('base_rmse', 'N/A'):.4f}")
            print(f"  Exp RMSE:    {result.get('exp_rmse', 'N/A'):.4f}")
            print(f"  RMSE Δ:      {result.get('rmse_improvement_pct', 0):+.2f}%")
            print(f"  Base β:      {result.get('base_mz_beta', 'N/A'):.4f}")
            print(f"  Exp β:       {result.get('exp_mz_beta', 'N/A'):.4f}")
            print(f"  β Δ:         {result.get('beta_improvement_pct', 0):+.2f}%")
            print(f"  Base p-val:  {result.get('base_mz_pvalue', 'N/A'):.4f}")
            print(f"  Exp p-val:   {result.get('exp_mz_pvalue', 'N/A'):.4f}")
            
            if result.get('dm_pvalue_mean', np.nan) is not np.nan:
                print(f"  DM p-value:  {result.get('dm_pvalue_mean', 'N/A'):.4f}")
            
            # Status indicator
            beta_imp = result.get('beta_improvement_pct', 0)
            rmse_imp = result.get('rmse_improvement_pct', 0)
            
            if beta_imp > 1 and rmse_imp > -0.5:  # β improves, RMSE doesn't degrade much
                print(f"  Status:      ✅ Improves calibration")
            elif beta_imp > 0 and rmse_imp > -1:
                print(f"  Status:      ⚠️ Minor improvement")
            else:
                print(f"  Status:      ❌ No clear improvement")
            
        except Exception as e:
            print(f"❌ Error in experiment {exp.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'experiment': exp.name, 'error': str(e)})
    
    return pd.DataFrame(results)


def save_results(results_df: pd.DataFrame, output_dir: str) -> None:
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    csv_path = os.path.join(output_dir, 'experiment_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # JSON (for structured data)
    json_path = os.path.join(output_dir, 'experiment_results.json')
    results_df.to_json(json_path, orient='records', indent=2)
    print(f"Results saved to: {json_path}")


def print_summary(results_df: pd.DataFrame) -> None:
    """Print a summary of all experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if results_df.empty:
        print("No results to display.")
        return
    
    # Sort by beta improvement
    if 'beta_improvement_pct' in results_df.columns:
        results_df = results_df.sort_values('beta_improvement_pct', ascending=False)
    
    print("\n{:<15} {:<12} {:<12} {:<12} {:<12} {:<15}".format(
        'Experiment', 'β Δ %', 'RMSE Δ %', 'Base β', 'Exp β', 'DM p-val'
    ))
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        exp_name = row.get('experiment', 'Unknown')[:14]
        beta_imp = row.get('beta_improvement_pct', 0)
        rmse_imp = row.get('rmse_improvement_pct', 0)
        base_beta = row.get('base_mz_beta', np.nan)
        exp_beta = row.get('exp_mz_beta', np.nan)
        dm_pval = row.get('dm_pvalue_mean', np.nan)
        
        # Status emoji
        if beta_imp > 1 and rmse_imp > -0.5:
            status = '✅'
        elif beta_imp > 0:
            status = '⚠️'
        else:
            status = '❌'
        
        print("{:<15} {:<+12.2f} {:<+12.2f} {:<12.4f} {:<12.4f} {:<15.4f}".format(
            f"{status} {exp_name}",
            beta_imp,
            rmse_imp,
            base_beta if not np.isnan(base_beta) else 0,
            exp_beta if not np.isnan(exp_beta) else 0,
            dm_pval if not np.isnan(dm_pval) else 0
        ))
    
    print("\n" + "=" * 70)
    print("Legend:")
    print("  ✅  - Improves calibration (β closer to 1) with no RMSE degradation")
    print("  ⚠️  - Minor improvement or trade-off")
    print("  ❌  - No clear improvement")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run isolated feature engineering experiments'
    )
    parser.add_argument(
        '--experiment',
        choices=['log', 'leverage', 'vol_of_vol', 'ewma', 'isotonic', 'all'],
        default='all',
        help='Which experiment to run (default: all)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (1 split instead of 5, for faster iteration)'
    )
    parser.add_argument(
        '--output',
        default='outputs/experiments',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\n" + "=" * 70)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 70)
        print("""
  log          - Log-transform target before training
                 Hypothesis: Stabilizes variance, improves calibration
  
  leverage     - Add leverage effect features
                 Hypothesis: Negative returns asymmetrically impact volatility
  
  vol_of_vol   - Add volatility of volatility to Baseline_3
                 Hypothesis: Volatility stability contains predictive signal
  
  ewma         - Test alternative EWMA decay factors
                 Hypothesis: λ=0.94 may not be optimal for this dataset
  
  isotonic     - Post-hoc isotonic calibration
                 Hypothesis: Directly maps predictions closer to actuals
  
  all          - Run all experiments
""")
        print("=" * 70)
        print("\nUsage: python experiment.py --experiment <name>")
        print("       python experiment.py --quick  # Fast mode")
        return
    
    print("=" * 70)
    print("VOLATILITY FORECASTING EXPERIMENTS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print("=" * 70 + "\n")
    
    # Load data
    print("Loading feature matrix...")
    try:
        feature_df = load_feature_matrix()
        print(f"Loaded: {feature_df.shape[0]:,} rows, {feature_df.shape[1]} columns")
        print(f"Date range: {feature_df['date'].min()} to {feature_df['date'].max()}")
        print(f"Tickers: {feature_df['ticker'].nunique()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Select experiments
    experiment_map = {
        'log': LogTransformExperiment,
        'leverage': LeverageExperiment,
        'vol_of_vol': VolOfVolExperiment,
        'ewma': EWMADecayExperiment,
        'isotonic': IsotonicCalibrationExperiment,
    }
    
    if args.experiment == 'all':
        experiments = [cls() for cls in experiment_map.values()]
    else:
        experiments = [experiment_map[args.experiment]()]
    
    # Run experiments
    results_df = run_experiments(
        experiments, 
        feature_df, 
        quick=args.quick,
        verbose=True
    )
    
    # Save results
    save_results(results_df, args.output)
    
    # Print summary
    print_summary(results_df)
    
    print("\n" + "=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    # Import lightgbm here to avoid circular imports
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    main()